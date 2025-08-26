import sys
import GEOparse

print(sys.argv[2:])

gse_list = sys.argv[2:]
output = sys.argv[1]

#Process data for every dataset
for gse_name in gse_list:
    print(gse_name)
    gse = GEOparse.get_GEO(geo=gse_name, destdir=output)