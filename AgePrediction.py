#Should probably change this to use the csv file not h5 data

import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import shap

from scipy.stats import ttest_1samp

#File paths
idmap_train_path = "trainmap.csv"
idmap_test_path = "testmap.csv"
train_path = "train.h5"
test_path = "test.h5"
siteList = "siteList.txt"
#Switching to Disease SHAP values shows similar MAE...
SHAPValues = "Age SHAP Values.txt"

#Number of top contributing features to be included in model
topN = 468


#Parameters for model
#Might need to change parameters
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
    "metric": "mae",   
}


#Load selected features
total_mean_SHAP_values = np.loadtxt(SHAPValues)
topNFeatures = np.argsort(total_mean_SHAP_values)[-topN:][::-1].tolist()
featureIndices = np.array(sorted(topNFeatures))

#load h5 data including selected top contributing features
def load_methylation_h5(path):
    methylation = h5py.File(path, "r")["data"]
    h5py.File(path, "r").close()
    return methylation[:,featureIndices]

#Load idmap containing ages of individuals
idmap = pd.read_csv(idmap_train_path, sep=",")
age = idmap.age.to_numpy()

#Load methylation dataset and test, split dataset into training and validation
print("Loading Data...")
start = time.time()
methylation = load_methylation_h5(train_path)
methylation_test = load_methylation_h5(test_path)
print(f"Loading time: {time.time() - start:.4f}s")

#Split data
indices = np.arange(len(age))
[indices_train, indices_valid, train, valid] = train_test_split(
    indices, age, test_size=0.3, shuffle=True
)
methylation_train, methylation_valid = (
    methylation[indices_train],
    methylation[indices_valid],
)
feature_size = methylation_train.shape[1]
print(feature_size)
del methylation

#Start training
model = LGBMRegressor(**params)
print("Start training...")
start = time.time()
model.fit(methylation_train, train)
print(f"Training time: {time.time() - start:.4f}s")

prediction = model.predict(methylation_valid)
mae = mean_absolute_error(prediction, valid)
print("Mean Absolute Error:" , mae)


#Generate SHAP
print("SHAP results")
explainer = shap.Explainer(model, methylation_train)
shap_values = explainer(methylation_test)

#Assign CpG site name to feature name
with open(siteList, "r") as f:
    row_names = np.array(f.read().splitlines())
shap_values.feature_names = row_names[featureIndices]

# Summary plot
shap.summary_plot(shap_values, methylation_test)

#Save results into csv file including CpG site, mean absolute value, standard deviation, and p-value
#Note p-value below 0.05 indicates mean absolute value is statistically significantly different from 0
shap_array = shap_values.values
mean_abs_shap = np.abs(shap_array).mean(axis=0)
std_shap = shap_array.std(axis=0)

p_values = []
for i in range(shap_array.shape[1]):
    _, p = ttest_1samp(shap_array[:, i], popmean=0)
    p_values.append(p)

SHAPResults = pd.DataFrame(
    {
        'Site Name':row_names[featureIndices],
        'Mean ABS SHAP Value':mean_abs_shap,
        'Stdev':std_shap,
        'p-value': p_values
    }
)
SHAPResults.to_csv('AgeResults.csv', index = False)