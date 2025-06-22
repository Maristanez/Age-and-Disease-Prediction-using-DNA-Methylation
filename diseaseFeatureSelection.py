import time
import random
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score
)
from lightgbm import LGBMClassifier
import shap

idmap_train_path = "disease_idmap.csv"
train_path = "disease_methylation_data.h5"

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

chunkSize = 111743

disease = "Alzheimer's disease"
control = 'control'
mci = "Mild Cognitive Impairment"

#Evaluation for classifier
def evaluation(y_valid, y_pred):
    print("F1 Score:", f1_score(y_valid, y_pred))
    print("Accuracy:", accuracy_score(y_valid, y_pred))
    print("Precision:", precision_score(y_valid, y_pred))
    print("Recall:", recall_score(y_valid, y_pred))

#load h5 data
def load_methylation_h5(path, i,sample_indices):
    with h5py.File(path, "r") as f:
        data = f["data"]
        methylation = data[sample_indices, i-chunkSize:i]
    return methylation

#load idmap
def load_idmap(idmap_dir, disease, control):
    idmap = pd.read_csv(idmap_dir, sep=",")
    mask = (idmap['disease_state'] == disease) | (idmap['disease_state'] == control)
    diseaseSelection = idmap[mask].copy()

    diseaseSelection['disease_state'] = diseaseSelection['disease_state'].replace({disease: 1, control: 0})
    disease_type = diseaseSelection.disease_state.to_numpy()
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
    shap_values = explainer(methylation_valid)
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

    #Evaluate model
    prediction = model.predict(methylation_valid)
    evaluation(y_valid, prediction)

    print("Getting indices of top N features")
    mean_abs_shap = mean_abs_shap = get_top_features(model, methylation_train)
    #Add it to total
    total_mean_SHAP_values = np.concatenate((total_mean_SHAP_values, mean_abs_shap))
    i += chunkSize

print("Saving NumPy Array")
print(total_mean_SHAP_values.size)
np.savetxt('Disease_SHAP_Values.txt', total_mean_SHAP_values)
