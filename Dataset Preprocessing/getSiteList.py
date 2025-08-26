import pandas as pd
import h5py
import numpy as np
import gc
import sys

gse_list = sys.argv[3:]
in_dir = sys.argv[2]
output = sys.argv[1]
methylation = None

print("Getting Site List")
for gse_name in gse_list:
    print(gse_name)
    gse_methylation = pd.read_csv(F"{in_dir}/{gse_name}_methylation.csv", index_col = 0, usecols=[0])
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
methylation.index.to_series().to_csv(output, index = False, header = False)