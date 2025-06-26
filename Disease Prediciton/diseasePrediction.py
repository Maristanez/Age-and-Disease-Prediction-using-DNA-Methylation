import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

from lightgbm import LGBMClassifier
import shap
import matplotlib.pyplot as plt

from scipy.stats import ttest_1samp

#File paths
idmap_train_path = "disease_idmap.csv"
train_path = "disease_methylation_data.h5"
siteList = "disease_CpG_sites.txt"
SHAPValues = "Disease_SHAP_Values.txt"

#Disease of interest to compare with control
disease = "Alzheimer's disease"
control = 'control'
mci = "Mild Cognitive Impairment"

#Number of top contributing features to be included in model
topN = 500

#Parameters for model
#Might need to change parameters
#Also consider imbalance between disease of interest and control
seed = int(time.time())
np.random.seed(seed)
random.seed(seed)
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "is_unbalance": True,
    'learning_rate': 0.09497756557572937, 
    'num_leaves': 54, 
    'max_depth': 4, 
    'feature_fraction': 0.20086507526204983, 
    'bagging_fraction': 0.555135932404743, 
    'bagging_freq': 3, 
    'lambda_l1': 1.660777046250498, 
    'lambda_l2': 3.3614276617667427
}

#Load selected features
total_mean_SHAP_values = np.loadtxt(SHAPValues)
topNFeatures = np.argsort(total_mean_SHAP_values)[-topN:][::-1].tolist()
featureIndices = np.array(sorted(topNFeatures))

#Evaluation for classifier
def evaluation(y_valid, y_pred, y_proba):
    itt = f"Top {topN}"
    rocauc = roc_auc_score(disease_type, y_proba)
    acc = accuracy_score(disease_type, y_pred)
    f1 = f1_score(disease_type, y_pred)
    pre = precision_score(disease_type, y_pred)
    rec = recall_score(disease_type, y_pred)

    print("F1 Score:", f1)
    print("Average CV Accuracy:", acc)
    print("Precision:", pre)
    print("Recall:", rec)

    # Save results
    evaluations = pd.read_csv('./Results/disease_evaluation_metrics.csv')
    finalResults = pd.DataFrame([{
        "Feature Chunks": itt,
        "AUC": rocauc,
        "F1 Score": f1,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec
    }])
    evaluations = pd.concat([evaluations, finalResults], ignore_index = True)
    evaluations.to_csv('./Results/disease_evaluation_metrics.csv', index=False)

    # Plot Confusion Matrix
    cm = confusion_matrix(y_valid, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Alzheimer's"]).plot()
    plt.savefig('./Results/confusion_matrix.png')
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(disease_type, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('./Results/ROC Curve.png')
    


#load h5 data for dataset
#Selects top contributing sites and samples containing disease of interest and control
def load_methylation_h5(path, sample_indices):
    with h5py.File(path, "r") as f:
        data = f["data"]
        methylation = data[sample_indices, :][: , featureIndices]
    return methylation

#load idmap
#Selects samples containing disease of interest or control
#Gets the disease type and the indices for loading h5 data of dataset
def load_idmap(idmap_dir, disease, control):
    idmap = pd.read_csv(idmap_dir, sep=",")
    mask = (idmap['disease_state'] == disease) | (idmap['disease_state'] == control)
    diseaseSelection = idmap[mask].copy()
    diseaseSelection['disease_state'] = diseaseSelection['disease_state'].replace({disease: 1, control: 0})
    disease_type = diseaseSelection.disease_state.to_numpy()
    selected_indices = idmap.index[mask].to_numpy()
    return disease_type, selected_indices


disease_type, disease_indices = load_idmap(idmap_train_path, disease, control)

#Load methylation dataset and test, split dataset into training and validation
print("Loading Data...")
start = time.time()
methylation = load_methylation_h5(train_path, disease_indices)
print(f"Loading time: {time.time() - start:.4f}s")

#Split data into training and validation
indices = np.arange(len(disease_type))

#Start training
print("Start training...")
model = LGBMClassifier(**params)

# Define stratified 5‑fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

#Evaluate model
model.fit(methylation, disease_type)
prediction = cross_val_predict(model, methylation, disease_type, cv=cv)
predictionA = cross_val_predict(model, methylation, disease_type, cv=cv, method="predict_proba")[:, 1]
evaluation(disease_type, prediction, predictionA)

#Generate SHAP
print("SHAP results")
#SHAP Explainer
explainer = shap.Explainer(model, methylation)
shap_values = explainer(methylation)

#Assign CpG site name to feature name
with open(siteList, "r") as f:
    feature_names = np.array(f.read().splitlines())
shap_values.feature_names = feature_names[featureIndices]

# Summary plot (global feature importance)
shap.summary_plot(shap_values, methylation, show = False)
plt.savefig('./Results/diseaseSHAP.png')
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
        'Site Name':feature_names[featureIndices],
        'Mean ABS SHAP Value':mean_abs_shap,
        'Stdev':std_shap,
        'p-value': p_values
    }
)

# Save top 20 CpG sites based on SHAP importance
top_20_SHAP = SHAPResults.sort_values(by='Mean ABS SHAP Value', ascending=False).head(20)
top_20_SHAP.to_csv("./top20s/disease_top_20_cpg_sites.csv", index=False)