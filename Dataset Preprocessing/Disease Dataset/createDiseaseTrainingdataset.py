import pandas as pd
import h5py
import numpy as np
import gc

gse_list = ["GSE144858","GSE153712"]
methylation = None


for gse_name in gse_list:
    print(gse_name)
    gse_methylation = pd.read_csv(F"{gse_name}_methylation.csv", index_col = 0)
    gse_methylation.astype("float32")
    print(gse_methylation.shape)
    if methylation is None:
        methylation = gse_methylation
    else:
        methylation = pd.concat([methylation,gse_methylation], axis = 1, join="outer")
    print(methylation.shape)
    del gse_methylation
    gc.collect()
print(methylation.head)
print("saving")
methylation_T = methylation.T
print(methylation_T.shape)
print("Final shape (samples, CpGs):", methylation_T.shape)

# Save to HDF5
with h5py.File("disease_methylation_data.h5", "w") as f:
    f.create_dataset("data", data=methylation_T.values, dtype='float32')
    f.create_dataset("row_names", data=np.array(methylation_T.index, dtype='S'))   # sample IDs
    f.create_dataset("col_names", data=np.array(methylation_T.columns, dtype='S')) # CpG IDs

print("Saved to disease_methylation_data.h5")

print("Getting Site List")
for gse_name in gse_list:
    print(gse_name)
    gse_methylation = pd.read_csv(F"{gse_name}_methylation.csv", index_col = 0, usecols=[0])
    print(gse_methylation)
    print(gse_methylation.shape)
    if methylation is None:
        methylation = gse_methylation
    else:
        methylation = pd.concat([methylation,gse_methylation], axis = 1, join="outer")
    print(methylation.shape)
    print(methylation.index)
    del gse_methylation
    gc.collect()
print(methylation.head)
print("saving")
methylation.index.to_series().to_csv("disease_CpG_sites.txt", index = False, header = False)