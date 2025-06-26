import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import (KFold, cross_val_score)
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp


#File paths
idmap_train_path = "age_idmap.csv"
train_path = "age_methylation_data.h5"
siteList = "age_CpG_sites.txt"
SHAPValues = "Age_SHAP_Values.txt"

#Number of top contributing features to be included in model
topN = 500

#Parameters for model
params = {
    'device' : 'gpu',
    'metric' : "mae",
    "boosting_type": "gbdt",
    'verbosity': -1,
    'learning_rate': 0.0331645648184633, 
    'num_leaves': 41, 
    'max_depth': 8, 
    'min_child_samples': 20, 
    'feature_fraction': 0.4671137310932305, 
    'bagging_fraction': 0.6826711995636782, 
    'bagging_freq': 1, 
    'lambda_l1': 1.2777036415584473, 
    'lambda_l2': 0.012062828849108644
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
# Load methylation data
print("Loading Data...")
start = time.time()
methylation = load_methylation_h5(train_path)
print(f"Loading time: {time.time() - start:.4f}s")
feature_size = methylation.shape[1]
print(f"Feature size: {feature_size}")

# Define model
model = LGBMRegressor(**params)

# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = -cross_val_score(model, methylation, age, cv=cv,
                              scoring='neg_mean_absolute_error')
medae_scores = -cross_val_score(model, methylation, age, cv=cv,
                                scoring='neg_median_absolute_error')

# Report mean scores
print("Cross-validated Mean Absolute Error:", np.mean(mae_scores))
print("Cross-validated Median Absolute Error:", np.mean(medae_scores))

# Save results
evaluations = pd.read_csv('Results/age_evaluation_metrics.csv')
finalResults = pd.DataFrame([{
    "Feature Chunks": f'Final {topN}',
    "Mean Absolute Error": np.mean(mae_scores),
    "Median Absolute Error": np.mean(medae_scores)
}])
evaluations = pd.concat([evaluations, finalResults], ignore_index=True)
evaluations.to_csv("Results/age_evaluation_metrics.csv", index=False)

#Create SHAP plot
model.fit(methylation, age)
explainer = shap.Explainer(model, methylation)
shap_values = explainer(methylation)

#Assign CpG site name to feature name
with open(siteList, "r") as f:
    row_names = np.array(f.read().splitlines())
shap_values.feature_names = row_names[featureIndices]

# Summary plot
shap.summary_plot(shap_values, methylation, show=False)
plt.savefig('./Results/ageSHAP.png')
plt.close()

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

# Save top 20 CpG sites based on SHAP importance
top_20_SHAP = SHAPResults.sort_values(by='Mean ABS SHAP Value', ascending=False).head(20)
top_20_SHAP.to_csv("./top20s/age_top_20_cpg_sites.csv", index=False)