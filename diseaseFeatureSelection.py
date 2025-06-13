import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#Probably need to change code since its changed to LGBMClassifier
from lightgbm import LGBMClassifier
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

chunkSize = 60689
methylationSiteCount = 485512

disease = "Alzheimer's disease"
control = 'control'

#load h5 data
#FIX THIS
def load_methylation_h5(path, i,sample_indices):
    with h5py.File(path, "r") as f:
        data = f["data"]
        if len(sample_indices) == 0:
            #For loading test dataset
            methylation = data[:, i-chunkSize:i]
        else:
            methylation = data[sample_indices, i-chunkSize:i]
    return methylation

#load idmap
def load_idmap(idmap_dir, disease, control):
    idmap = pd.read_csv(idmap_dir, sep=",")
    mask = (idmap['disease'] == disease) | (idmap['disease'] == control)
    diseaseSelection = idmap[mask].copy()

    diseaseSelection['disease'] = diseaseSelection['disease'].replace({disease: 1, control: 0})
    disease_type = diseaseSelection.disease.to_numpy()
    selected_indices = idmap.index[mask].to_numpy()
    return disease_type, selected_indices

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


disease_type, disease_indices = load_idmap(idmap_train_path, disease, control)

total_mean_SHAP_values = np.array([])
#485512
i = chunkSize
while i <= chunkSize*8:
    print(f"Feature Range: {i-chunkSize} - {i}")
    print("Loading Data...")
    start = time.time()
    methylation = load_methylation_h5(train_path,i,disease_indices)
    #print(methylation)
    methylation_test = load_methylation_h5(test_path,i,[])
    print(f"Loading time: {time.time() - start:.4f}s")

    indices = np.arange(len(disease_type))
    methylation_train, methylation_valid, y_train, y_valid = split_training(indices, disease_type)
    feature_size = methylation_train.shape[1]

    del methylation

    model = LGBMClassifier(**params)
    print("Start training...")
    start = time.time()
    model.fit(methylation_train, y_train)
    print(f"Training time: {time.time() - start:.4f}s")
    
    print("Getting indices of top N features")
    explainer = shap.Explainer(model, methylation_train)
    shap_values = explainer(methylation_test)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    total_mean_SHAP_values = np.concatenate((total_mean_SHAP_values, mean_abs_shap))
    i += chunkSize

print("Saving NumPy Array")
print(total_mean_SHAP_values.size)
np.savetxt('Disease SHAP Values.txt', total_mean_SHAP_values)
