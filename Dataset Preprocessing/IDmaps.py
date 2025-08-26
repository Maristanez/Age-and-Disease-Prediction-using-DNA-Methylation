import sys
import pandas as pd
import GEOparse

print(sys.argv[3:])

gse_list = sys.argv[3:]
filePath = sys.argv[2]
output = sys.argv[1]

#Process data for every dataset
for gse_name in gse_list:
    print(gse_name)
    gse = GEOparse.get_GEO(filepath=f"{filePath}/{gse_name}_family.soft.gz")
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

                #Save value
                if key in ["sex", "gender", "Sex", "gender (1=woman; 0=man)"]:
                    metadata["sex"] = value
                elif key in ["age", "age (y)", "age in 2011"]:
                    metadata["age"] = value
                elif key in ["disease status", "disease state"]:
                    if value in ["Alzheimer's Disease", "Alzheimer's disease"]:
                        metadata["disease_state"] = "Alzheimer's"
                    elif value in ["Mild Cognitive Impairment", "mild cognitive impairment"]:
                        metadata["disease_state"] = "MCI"
                    elif value in ["healthy control", "control"]:
                        metadata["disease_state"] = "control"

        records.append(metadata)


    # Convert to DataFrame and save as csv file
    df = pd.DataFrame(records)
    print(df.head())
    df.to_csv(f'{output}/{gse_name}_idmap.csv', index = False)