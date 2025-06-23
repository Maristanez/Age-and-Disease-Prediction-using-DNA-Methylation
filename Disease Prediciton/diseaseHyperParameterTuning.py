import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import optuna
from sklearn.metrics import roc_auc_score


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

def objective(trial):
    # Define hyperparameters to tune
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "is_unbalance": True,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.9),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
    }


    model = LGBMClassifier(**params)
    model.fit(methylation_train, y_train)

    y_proba = model.predict_proba(methylation_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_proba)
    return auc

disease_type, disease_indices = load_idmap(idmap_train_path, disease, control)
#Load methylation dataset and test, split dataset into training and validation
print("Loading Data...")
start = time.time()
methylation = load_methylation_h5(train_path, disease_indices)
print(f"Loading time: {time.time() - start:.4f}s")

#Split data into training and validation
indices = np.arange(len(disease_type))

methylation_train, methylation_valid, y_train, y_valid = split_training(indices, disease_type)
del methylation


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000)

print("Best AUC:", study.best_value)
print("Best Params:", study.best_params)