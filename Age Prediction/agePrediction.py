import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import (mean_absolute_error, median_absolute_error)
import shap
import matplotlib as plt
from scipy.stats import ttest_1samp

#File paths
idmap_train_path = "age_idmap.csv"
train_path = "age_methylation_data.h5"
siteList = "age_CpG_sites.txt"
SHAPValues = "Age_SHAP_Values.txt"

#Number of top contributing features to be included in model
topN = 500


#Parameters for model
#Might need to change parameters
seed = int(time.time())
np.random.seed(seed)
random.seed(seed)
params = {
    'device' : 'gpu',
    'metric' : "mae",
    "boosting_type": "gbdt",
    'verbosity': -1,
    'learning_rate': 0.04048392049598949,
    'num_leaves': 29, 
    'max_depth': 6, 
    'min_child_samples': 26, 
    'feature_fraction': 0.4430238083554771, 
    'bagging_fraction': 0.711748619719678, 
    'bagging_freq': 4, 
    'lambda_l1': 0.06539240299243669, 
    'lambda_l2': 0.0018043222664827436
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

#Evaluations
prediction = model.predict(methylation_valid)

itt = f'Final {topN}'
mae = mean_absolute_error(prediction, valid)
medae = median_absolute_error(prediction, valid)
print("Mean Absolute Error:" , mae)
print("Median Absolute Error", medae)
evaluations = pd.read_csv('/Results/age_evaluation_metrics.csv')
finalResults = pd.DataFrame([{
    "Feature Chunks": itt,
    "Mean Absolute Error": mae,
    "Median Absolute Error": medae
    }])
evaluations = pd.concat([evaluations, finalResults], ignore_index = True)
evaluations.to_csv("./Results/disease_evaluation_metrics.csv", index = False)

#Generate SHAP
print("SHAP results")
explainer = shap.Explainer(model, methylation_train)
shap_values = explainer(methylation_valid)

#Assign CpG site name to feature name
with open(siteList, "r") as f:
    row_names = np.array(f.read().splitlines())
shap_values.feature_names = row_names[featureIndices]

# Summary plot
shap.summary_plot(shap_values, methylation_valid)
plt.savefig('./Results/ageSHAP.png')

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
#SHAPResults.to_csv('AgeResults.csv', index = False)

# Save top 20 CpG sites based on SHAP importance
top_20_SHAP = SHAPResults.sort_values(by='Mean ABS SHAP Value', ascending=False).head(20)
top_20_SHAP.to_csv("./top20s/age_top_20_cpg_sites.csv", index=False)