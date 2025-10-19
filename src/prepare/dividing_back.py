import pandas as pd
import os

all_pkl = "C:\\Devign\\devign\\data\\cpg\\all_cpg.pkl"
df = pd.read_pickle(all_pkl)

cpg_dir = "C:\\Devign\\devign\\data\\cpg\\per_file\\"
os.makedirs(cpg_dir, exist_ok=True)

for file_idx, group in df.groupby("file"):
    out_file = os.path.join(cpg_dir, f"{file_idx}_cpg.pkl")
    group.to_pickle(out_file)
    print(f"Saved {out_file} ({len(group)} functions)")
