import GEOparse
import pandas as pd

gse_list = ["GSE153712"]

#Process data for every dataset
for gse_name in gse_list:
    gse = GEOparse.get_GEO(gse_name, destdir=".")

    # List to hold the extracted rows
    records = []

    # Iterate through all samples
    for gsm_name, gsm in gse.gsms.items():
        metadata = {
                "sample_id": gsm_name,
                "sex": None,
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
                elif key == "disease status":
                    metadata["disease_state"] = value

        records.append(metadata)


    # Convert to DataFrame and save as csv file
    df = pd.DataFrame(records)
    print(df.head())

    df.to_csv(f'{gse_name}_idmap.csv', index = False)