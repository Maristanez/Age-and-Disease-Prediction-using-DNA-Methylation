import GEOparse
import pandas as pd

# Load the GSE dataset
gse_name = "GSE144858"
gse = GEOparse.get_GEO(gse_name, destdir="./Dataset Preprocessing/Disease Dataset/SOFT Files")


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
            "disease_state": None,
        }

        # Go through the sample's characteristics list
    for item in gsm.metadata.get("characteristics_ch1", []):
        if ":" in item:
            key, value = item.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "sex":
                metadata["sex"] = value
            elif key == "age":
                metadata["age"] = value
            elif key == "disease state":
                metadata["disease_state"] = value

    records.append(metadata)

    print(gsm.table.head())
    values = pd.DataFrame({gsm_name : gsm.table["VALUE"]})
    print(values.head)
    df_methylation[gsm_name] = values

# Convert to DataFrame and save
df = pd.DataFrame(records)
print(df.head())
print(df_methylation.head())

df.to_csv('./Dataset Preprocessing/Disease Dataset/ID Maps/GSE144858_idmap.csv', index = False)
df_methylation.to_csv('./Dataset Preprocessing/Disease Dataset/Methylation Files/GSE144858_methylation.csv', index = False)