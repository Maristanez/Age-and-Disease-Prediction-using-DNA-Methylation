import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import shap

idmap_train_path = "trainmap.csv"
idmap_test_path = "testmap.csv"
train_path = "train.h5"
test_path = "test.h5"

seed = int(time.time())
np.random.seed(seed)
random.seed(seed)

params = {
    'device': 'gpu',
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": -1,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample_for_bin": 200000,
    "objective": None,
    "class_weight": None,
    "min_split_gain": 0.0,
    "min_child_weight": 0.001,
    "min_child_samples": 20,
    "subsample": 1.0,
    "subsample_freq": 0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": seed,
    "n_jobs": None,
    "importance_type": "split",
}

chunk_size = 60689
#485512 total sites in dataset

#load h5 data
def load_methylation_h5(path,i):
    methylation = h5py.File(path, "r")["data"]
    h5py.File(path, "r").close()
    return methylation[:, i-chunk_size: i]

#load idmap
def load_idmap(idmap_dir):
    idmap = pd.read_csv(idmap_dir, sep=",")
    age = idmap.age.to_numpy()
    return age

#Split dataset using indices of age from the idmap
def split_training(indices,y):
    [indices_train, indices_valid, y_train, y_valid] = train_test_split(
        indices, y, test_size=0.3, shuffle=True
    )
    methylation_train, methylation_valid = (
        methylation[indices_train],
        methylation[indices_valid],
    )
    return methylation_train, methylation_valid, y_train, y_valid

def get_top_features(model, methylation_train):
    explainer = shap.Explainer(model, methylation_train)
    shap_values = explainer(methylation_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    return mean_abs_shap

age = load_idmap(idmap_train_path)

total_mean_SHAP_values = np.array([])
#485512
i = chunk_size
while i <= chunk_size*8:
    print(f"Feature Range: {i-chunk_size} - {i}")
    print("Loading Data...")
    start = time.time()
    methylation = load_methylation_h5(train_path,i)
    methylation_test = load_methylation_h5(test_path,i)
    print(f"Loading time: {time.time() - start:.4f}s")

    indices = np.arange(len(age))
    #Split dataset
    methylation_train, methylation_valid, y_train, y_valid = split_training(indices,age)
    del methylation

    print("Start training...")
    start = time.time()
    model = LGBMRegressor(**params)
    model.fit(methylation_train, y_train)
    print(f"Training time: {time.time() - start:.4f}s")

    prediction = model.predict(methylation_valid)
    mae = mean_absolute_error(prediction, y_valid)
    print("Mean Absolute Error:" , mae)
    
    print("Getting indices of top N features")
    mean_abs_shap = get_top_features(model, methylation_train)
    #Add it to total
    total_mean_SHAP_values = np.concatenate((total_mean_SHAP_values, mean_abs_shap))

    i += chunk_size

print("Saving NumPy Array")
print(total_mean_SHAP_values.size)
np.savetxt('Age SHAP Values.txt', total_mean_SHAP_values)