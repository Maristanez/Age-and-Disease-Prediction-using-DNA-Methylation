import pandas as pd

gse_list = ["GSE51032","GSE73103", "GSE64495", "GSE42861", "GSE40279", "GSE69270", "GSE41169", "GSE30870(edited)", "GSE144858"]
idmap = pd.DataFrame()

for gse_name in gse_list:
    print(gse_name)
    gse_idmap = pd.read_csv(F"f'./Dataset Preprocessing/Age Dataset/ID Maps/{gse_name}_idmap.csv'")
    gse_idmap["series_id"] = gse_name
    idmap = pd.concat([idmap, gse_idmap])

idmap.to_csv("age_idmap.csv", index = False)
