import numpy as np
import pandas as pd
from glob import glob


table = pd.DataFrame()
files = glob("*.npz")
for f in files:
    npz = np.load(f)
    nv = npz["nv_target"]
    score = npz["best_score"] * nv[0,1]
    table.loc[f.split(".c")[0], "L2"] = score

import pdb; pdb.set_trace()
table.to_csv("aggreg.csv", index=False)