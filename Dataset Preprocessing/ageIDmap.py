import pandas as pd
import sys

gse_list = sys.argv[3:]
in_dir = sys.argv[2]
output = sys.argv[1]

idmap = pd.DataFrame()

for gse_name in gse_list:
    print(gse_name)
    gse_idmap = pd.read_csv(F"{in_dir}/{gse_name}_idmap.csv")
    gse_idmap["series_id"] = gse_name
    idmap = pd.concat([idmap, gse_idmap])

idmap.to_csv(output, index = False)