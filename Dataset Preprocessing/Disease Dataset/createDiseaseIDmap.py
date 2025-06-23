import pandas as pd

gse_list = ["GSE144858","GSE153712"]
idmap = pd.DataFrame()

for gse_name in gse_list:
    print(gse_name)
    gse_idmap = pd.read_csv(F"./Dataset Preprocessing/Disease Dataset/ID Maps/{gse_name}_idmap.csv")
    gse_idmap["disease_state"] = gse_idmap["disease_state"].replace({
    "healthy control": "control",
    "mild cognitive impairment": "Mild Cognitive Impairment",
    
})
    gse_idmap["series_id"] = gse_name
    idmap = pd.concat([idmap, gse_idmap])

idmap.to_csv("disease_idmap.csv", index = False)
