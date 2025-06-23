import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# File paths
h5_file = "age_methylation_data.h5"
site_list_file = "age_CpG_sites.txt"
top_sites_file = "./top20s/age_top_20_cpg_sites.csv"
disease_top_sites_file = "./top20s/disease_top_20_cpg_sites.csv"
idmap_file = "age_idmap.csv"

# Load CpG site names in the same order as H5 file
with open(site_list_file) as f:
    all_cpg_sites = np.array(f.read().splitlines())

# Create mapping from CpG name to index
cpg_to_index = {name: i for i, name in enumerate(all_cpg_sites)}

# Load age
idmap = pd.read_csv(idmap_file)
ages = idmap["age"].values.astype(float)

# ---- Age CpG Sites ----
top_cpg_df = pd.read_csv(top_sites_file)
top_cpg_names = top_cpg_df["Site Name"].tolist()
top_indices = np.array(sorted([cpg_to_index[cpg] for cpg in top_cpg_names]))

# Load methylation data for age CpGs
with h5py.File(h5_file, "r") as f:
    age_data = f["data"][:, top_indices]

# Compute correlation and plot for age CpGs
results = []
plt.figure(figsize=(20, 25))
for i, cpg in enumerate(top_cpg_names):
    values = age_data[:, i].astype(float)
    valid = ~np.isnan(values) & ~np.isnan(ages)
    if np.sum(valid) > 1 and np.std(values[valid]) > 0 and np.std(ages[valid]) > 0:
        corr, p = pearsonr(values[valid], ages[valid])
    else:
        corr, p = np.nan, np.nan

    results.append({"CpG Site": cpg, "Pearson r": corr, "p-value": p})


    # --- Individual plot ---
    plt_indiv = plt.figure(figsize=(6, 4))
    sns.regplot(x=ages[valid], y=values[valid], scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title(f"{cpg}\nr = {corr:.2f}, p = {p:.2e}" if not np.isnan(corr) else f"{cpg}\nInvalid correlation")
    plt.ylabel("Methylation Beta Value")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(f"Results/CpG Age Correlations/{cpg}.png", dpi=300)
    plt.close()

    # --- Grouped subplot ---
    plt.subplot(5, 4, i + 1)
    sns.regplot(x=ages[valid], y=values[valid], scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.title(f"{cpg}\nr = {corr:.2f}, p = {p:.2e}" if not np.isnan(corr) else f"{cpg}\nInvalid correlation")
    plt.ylabel("Methylation Beta Value")
    plt.xlabel("Age")

plt.tight_layout()
plt.savefig("Results/age_top20_CpG_correlations.png", dpi=300)
plt.show()

# Save correlation results
pd.DataFrame(results).to_csv("Results/age_top_20_CpG_correlation_stats.csv", index=False)

# ---- Disease CpG Sites ----
disease_df = pd.read_csv(disease_top_sites_file)
disease_cpg_names = disease_df["Site Name"].tolist()

# Filter CpGs that are not in age data
valid_disease_cpgs = [cpg for cpg in disease_cpg_names if cpg in cpg_to_index]
disease_indices = np.array(sorted([cpg_to_index[cpg] for cpg in valid_disease_cpgs]))

# Load methylation data for valid disease CpGs
with h5py.File(h5_file, "r") as f:
    disease_data = f["data"][:, disease_indices]

# Compute correlations and plot scatter+regression lines
disease_results = []
plt.figure(figsize=(20, 25))

for i, cpg in enumerate(valid_disease_cpgs):
    values = disease_data[:, i].astype(float)

    # Define valid mask: ignore NaNs, but keep age = 0
    mask = ~np.isnan(values) & ~np.isnan(ages)
    x = ages[mask]
    y = values[mask]

    # Correlation
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        corr, p = pearsonr(y, x)
    else:
        corr, p = np.nan, np.nan

    disease_results.append({
        "CpG Site": cpg,
        "Pearson r": corr,
        "p-value": p
    })

    # Plot
    # ---- Individual Figure ----
    plt_indiv = plt.figure(figsize=(6, 4))
    if len(x) > 0:
        sns.regplot(x=x, y=y, scatter_kws={'s': 10}, line_kws={'color': 'blue'})
        plt.title(f"{cpg}\nr = {corr:.2f}, p = {p:.2e}" if not np.isnan(corr) else f"{cpg}\nInvalid correlation")
    else:
        plt.title(f"{cpg}\nNo valid data")
    plt.xlabel("Age")
    plt.ylabel("Methylation Beta Value")
    plt.tight_layout()
    plt.savefig(f"Results/CpG Disease Correlations/{cpg}.png", dpi=300)
    plt.close()

    # ---- Grouped Subplot ----
    plt.subplot(5, 4, i + 1)
    if len(x) > 0:
        sns.regplot(x=x, y=y, scatter_kws={'s': 10}, line_kws={'color': 'blue'})
        plt.title(f"{cpg}\nr = {corr:.2f}, p = {p:.2e}" if not np.isnan(corr) else f"{cpg}\nInvalid correlation")
    else:
        plt.title(f"{cpg}\nNo valid data")
    plt.xlabel("Age")
    plt.ylabel("Methylation Beta Value")
    
plt.tight_layout()
plt.savefig("Results/disease_top20_CpG_correlations.png", dpi=300)
plt.show()

# Save correlation results for disease CpGs
pd.DataFrame(disease_results).to_csv("Results/disease_top_20_CpG_correlation_stats.csv", index=False)
