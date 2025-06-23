import GEOparse
import pandas as pd

gse_list = ["GSE51032","GSE73103", "GSE64495", "GSE42861", "GSE40279", "GSE69270", "GSE41169", "GSE30870", "GSE144858"]

#Process data for every dataset
for gse_name in gse_list:
    gse = GEOparse.get_GEO(gse_name, destdir=".")

    #Get CpG site names
    for gsm_name, gsm in gse.gsms.items():
        cpg_sites = gsm.table["ID_REF"]
        break
    df_methylation = pd.DataFrame({'CpG Sites' : cpg_sites})
    print(df_methylation.head())

    # List to hold the extracted rows
    records = []

    # Iterate through all samples
    for gsm_name, gsm in gse.gsms.items():
        metadata = {
                "sample_id": gsm_name,
                "sex": None,
                "age": None,
            }

            # Go through the sample's characteristics list
        for item in gsm.metadata.get("characteristics_ch1", []):
            if ":" in item:
                key, value = item.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                #Save value
                if key in ["sex", "gender", "Sex", "gender (1=woman; 0=man)"]:
                    metadata["sex"] = value
                elif key in ["age", "age (y)", "age in 2011"]:
                    metadata["age"] = value

        records.append(metadata)

        values = pd.DataFrame({gsm_name : gsm.table["VALUE"]})
        df_methylation[gsm_name] = values

    # Convert to DataFrame and save as csv file
    df = pd.DataFrame(records)
    print(df.head())
    print(df_methylation.head())

    df.to_csv(f'{gse_name}_idmap.csv', index = False)
    df_methylation.to_csv(f'{gse_name}_methylation.csv', index = False)

import pandas as pd
import gc

gse_list = ["GSE51032","GSE73103", "GSE64495", "GSE42861", "GSE40279", "GSE69270", "GSE41169", "GSE30870", "GSE144858"]
methylation = None
print(gse_list[0:9])


for gse_name in gse_list[0:9]:
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
methylation.index.to_series().to_csv("age_CpG_sites.txt", index = False, header = False)
