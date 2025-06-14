import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
import shap
import matplotlib.pyplot as plt

from scipy.stats import ttest_1samp

#File paths
idmap_train_path = "trainmap.csv"
idmap_test_path = "testmap.csv"
train_path = "train.h5"
test_path = "test.h5"
siteList = "siteList.txt"
#Switching to Age SHAP values shows similar MAE...
SHAPValues = "Disease SHAP Values.txt"

#Disease of interest to compare with control
disease = "Alzheimer's disease"
control = 'control'

#Number of top contributing features to be included in model
topN = 650


#Parameters for model
#Might need to change parameters
#Also consider imbalance between disease of interest and control
seed = int(time.time())
np.random.seed(seed)
random.seed(seed)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',           # or 'regression' if predicting age
    'metric': 'binary_logloss',     # or 'auc' if classification
    'learning_rate': 0.01,          # smaller to reduce overfitting
    'num_leaves': 8,                # keep small due to low sample size
    'max_depth': 4,                 # shallow trees help generalize
    'min_data_in_leaf': 20,         # reduce overfitting on noise
    'feature_fraction': 0.05,       # randomly sample 5% of features per tree
    'bagging_fraction': 0.8,        # randomly sample 80% of rows
    'bagging_freq': 1,              # perform bagging every tree
    'lambda_l1': 1.0,               # L1 regularization (feature selection)
    'lambda_l2': 1.0,               # L2 regularization
    'verbosity': -1,
    'is_unbalance': True            # use if class imbalance is present
}

#Load selected features
total_mean_SHAP_values = np.loadtxt(SHAPValues)
topNFeatures = np.argsort(total_mean_SHAP_values)[-topN:][::-1].tolist()
featureIndices = np.array(sorted(topNFeatures))

#Evaluation for classifier
def evaluation(y_valid, y_pred, y_proba):
    print("AUC-ROC:", roc_auc_score(y_valid, y_proba))
    print("F1 Score:", f1_score(y_valid, y_pred))
    print("Accuracy:", accuracy_score(y_valid, y_pred))
    print("Precision:", precision_score(y_valid, y_pred))
    print("Recall:", recall_score(y_valid, y_pred))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_valid, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

#load h5 data for dataset
#Selects top contributing sites and samples containing disease of interest and control
def load_methylation_h5(path, sample_indices):
    with h5py.File(path, "r") as f:
        data = f["data"]
        if len(sample_indices) == 0:
            #For loading test dataset
            methylation = data[:,featureIndices]
        else:
            methylation = data[sample_indices, :][: , featureIndices]
    return methylation

#load idmap
#Selects samples containing disease of interest or control
#Gets the disease type and the indices for loading h5 data of dataset
def load_idmap(idmap_dir, disease, control):
    idmap = pd.read_csv(idmap_dir, sep=",")
    mask = (idmap['disease'] == disease) | (idmap['disease'] == control)
    diseaseSelection = idmap[mask].copy()
    diseaseSelection['disease'] = diseaseSelection['disease'].replace({disease: 1, control: 0})
    disease_type = diseaseSelection.disease.to_numpy()
    selected_indices = idmap.index[mask].to_numpy()
    return disease_type, selected_indices

#Split dataset using indices of disease from the idmap
def split_training(indices,y):
    [indices_train, indices_valid, y_train, y_valid] = train_test_split(
        indices, y, test_size=0.3, shuffle=True
    )
    methylation_train, methylation_valid = (
        methylation[indices_train],
        methylation[indices_valid],
    )
    return methylation_train, methylation_valid, y_train, y_valid


disease_type, disease_indices = load_idmap(idmap_train_path, disease, control)

#Load methylation dataset and test, split dataset into training and validation
print("Loading Data...")
start = time.time()
methylation = load_methylation_h5(train_path, disease_indices)
methylation_test = load_methylation_h5(test_path, [])
print(f"Loading time: {time.time() - start:.4f}s")

#Split data into training and validation
indices = np.arange(len(disease_type))

methylation_train, methylation_valid, y_train, y_valid = split_training(indices, disease_type)
del methylation

#Start training
print("Start training...")
model = LGBMClassifier(**params)
start = time.time()
model.fit(methylation_train, y_train)
print(f"Training time: {time.time() - start:.4f}s")

prediction = model.predict(methylation_valid)
predictionA = model.predict_proba(methylation_valid)[:, 1]

#Evaluate model
evaluation(y_valid, prediction, predictionA)

#Generate SHAP
print("SHAP results")
#SHAP Explainer
explainer = shap.Explainer(model, methylation_train)
shap_values = explainer(methylation_test)

#Assign CpG site name to feature name
with open(siteList, "r") as f:
    row_names = np.array(f.read().splitlines())
shap_values.feature_names = row_names[featureIndices]

# Summary plot (global feature importance)
shap.summary_plot(shap_values, methylation_test)
shap.plots.bar(shap.Explanation(values=shap_values[1], data=methylation_test, feature_names=row_names[featureIndices]))

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
SHAPResults.to_csv('diseaseResults.csv', index = False)