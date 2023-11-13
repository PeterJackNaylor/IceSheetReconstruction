import pandas as pd
from glob import glob

files = []
for f in glob("*.csv"):
    files.append(pd.read_csv(f, index_col=0))
tables = pd.concat(files, axis=1)
tables.columns = list(range(tables.shape[1]))

tables.to_csv("performance.tsv", sep="\t")
