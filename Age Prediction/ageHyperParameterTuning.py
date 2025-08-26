import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from lightgbm import LGBMRegressor
import optuna

# Setup
idmap_train_path = "./Data Preprocessing/Age/age_idmap.csv"
train_path = "./Data Preprocessing/Age/age_methylation_data.h5"
SHAPValues = "Age_SHAP_Values.txt"
topN = 468

# Reproducibility
seed = int(time.time())
np.random.seed(seed)
random.seed(seed)

# Load age labels
idmap = pd.read_csv(idmap_train_path)
ages = idmap["age"].values.astype(float)

# Load top SHAP feature indices
shap_values = np.loadtxt(SHAPValues)
top_indices = np.argsort(shap_values)[-topN:][::-1]
featureIndices = np.sort(top_indices)

# Load methylation data (only once)
def load_methylation(path, indices):
    with h5py.File(path, "r") as f:
        return np.nan_to_num(f["data"][:, indices], nan=0.0)

print("Loading methylation data...")
methylation_data = load_methylation(train_path, featureIndices)
print("Shape:", methylation_data.shape)

# Split data
indices = np.arange(len(ages))
idx_train, idx_valid, y_train_full, y_valid = train_test_split(
    indices, ages, test_size=0.3, random_state=42
)
X_train_full = methylation_data[idx_train]
X_valid = methylation_data[idx_valid]

# Optuna objective
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10, log=True),
        'n_estimators': 200,
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'device': 'gpu'
    }

    model = LGBMRegressor(**params)
    scores = cross_val_score(
        model, X_train_full, y_train_full,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=3
    )
    return -scores.mean()

# Run Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout = 1800)

print("Best Parameters:", study.best_params)
print("Best MAE:", study.best_value)
