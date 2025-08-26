import sys
import pandas as pd
import GEOparse

gse_list = sys.argv[3:]
filePath = sys.argv[2]
output = sys.argv[1]

# Load the GSE dataset
gse_name = sys.argv[1]
for gse_name in gse_list:
    print(gse_name)
    gse = GEOparse.get_GEO(filepath=f"{filePath}/{gse_name}_family.soft.gz")

    #Get CpG site names
    for gsm_name, gsm in gse.gsms.items():
        cpg_sites = gsm.table["ID_REF"]
        break
    df_methylation = pd.DataFrame({'CpG Sites' : cpg_sites})
    print(df_methylation.head())


    # Iterate through all samples
    for gsm_name, gsm in gse.gsms.items():
        print(gsm.table.head())
        values = pd.DataFrame({gsm_name : gsm.table["VALUE"]})
        print(values.head)
        df_methylation[gsm_name] = values

    # Convert to DataFrame and save
    df_methylation.to_csv(f'{output}/{gse_name}_methylation.csv', index = False)