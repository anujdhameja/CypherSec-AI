# import json
# import pandas as pd
# import glob
# import os

# json_dir = "data/cpg/"
# pkl_dir = "data/cpg/pkl/"

# os.makedirs(pkl_dir, exist_ok=True)

# for json_file in glob.glob(os.path.join(json_dir, "*_cpg.json")):
#     with open(json_file, "r") as f:
#         data = json.load(f)
    
#     functions_list = data.get("functions", [])
#     if not functions_list:
#         continue  # skip if empty

#     df = pd.DataFrame(functions_list)
#     file_name = os.path.basename(json_file).replace(".json", ".pkl")
#     df.to_pickle(os.path.join(pkl_dir, file_name))

#     print(f"Saved {file_name}")


import json
import pandas as pd
import os

# Paths
json_folder = "data/cpg/"
pkl_folder = "data/cpg/"

# Make sure pkl folder exists
os.makedirs(pkl_folder, exist_ok=True)

# List all JSON files
json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

for json_file in sorted(json_files):
    json_path = os.path.join(json_folder, json_file)
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Extract functions
    funcs = data.get("functions", [])
    
    # Skip if no functions found
    if not funcs:
        print(f"{json_file} has no functions, skipping...")
        continue
    
    # Create DataFrame
    df = pd.DataFrame({
        "func": funcs,
        "target": [0]*len(funcs),  # Replace 0 with actual labels if available
        "Index": range(len(funcs)),
        "cpg": [None]*len(funcs)
    })
    
    # Save as .pkl
    base_name = os.path.splitext(json_file)[0]
    pkl_path = os.path.join(pkl_folder, f"{base_name}.pkl")
    df.to_pickle(pkl_path)
    
    print(f"Saved {pkl_path} with {len(df)} functions")
