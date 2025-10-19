# # import pandas as pd

# # df = pd.read_pickle("data/cpg/0_cpg.pkl")
# # print(df.head())
# # print(df.shape)
# # import json

# # with open("data/cpg/0_cpg.json") as f:
# #     data = json.load(f)

# # print(len(data))          # Number of function entries
# # print(data[0].keys())     # Keys of the first function



# import json

# with open("data/cpg/0_cpg.json") as f:
#     data = json.load(f)

# # Print top-level type
# print(type(data))

# # If it's a dict, print its keys
# if isinstance(data, dict):
#     print("Top-level keys:", list(data.keys()))

# # If it’s a list, print first item keys
# elif isinstance(data, list) and len(data) > 0:
#     print("First item keys:", data[0].keys())

# # Also print length
# print("Number of top-level entries:", len(data))


# import pandas as pd

# df = pd.read_pickle("data/cpg/0_cpg.pkl")
# print(df.head())
# print(df.columns)
# print(len(df))


# import json

# with open("data/cpg/0_cpg.json") as f:
#     data = json.load(f)

# print(data.keys())      # should show ['functions']
# print(len(data['functions']))


# import pandas as pd
# df = pd.read_pickle("data/cpg/36_cpg.pkl")
# print(df['func'].head(10))
# print(type(df['func'].iloc[0]))


# import pandas as pd
# df = pd.read_pickle("data/cpg/36_cpg.pkl")
# print(df.head())
# print(df['cpg'].isnull().sum())


# import pickle, os

# file = r"C:\Devign\devign\data\cpg\1_cpg.pkl"   # adjust PATH
# # with open(file, "rb") as f:
# #     df = pickle.load(f)

# # print(df.head())
# # print(df.info())
# # print("Any non-null cpg?", df['cpg'].notnull().any())

# import os, pickle, pandas as pd

# path = r"C:\Devign\devign\data\cpg"

# summary = []
# for fname in sorted(os.listdir(path)):
#     if fname.endswith(".pkl"):
#         fpath = os.path.join(path, fname)
#         try:
#             with open(fpath, "rb") as f:
#                 df = pickle.load(f)
#             rows = len(df)
#         except Exception as e:
#             rows = f"Error: {e}"
#         summary.append((fname, rows))

# print(pd.DataFrame(summary, columns=["File", "Rows"]))


# import json, os

# path = r"C:\Devign\devign\data\cpg\1_cpg.json"
# print("File size:", os.path.getsize(path), "bytes")

# with open(path, "r", encoding="utf-8") as f:
#     try:
#         data = json.load(f)
#         print("Keys:", data.keys())
#         print("First node sample:", data["nodes"][0] if "nodes" in data and data["nodes"] else "No nodes")
#     except Exception as e:
#         print("Error reading JSON:", e)



# import pandas as pd
# import os

# cpg_dir = "C:\\Devign\\devign\\data\\cpg\\per_file\\"
# for f in os.listdir(cpg_dir):
#     if f.endswith(".pkl"):
#         df = pd.read_pickle(os.path.join(cpg_dir, f))
#         print(f"{f}: {len(df)} functions, columns={df.columns.tolist()}")
#         print("Sample:", df.iloc[0]["func"])



# import pickle
# with open("C:/Devign/devign/data/cpg/per_file/2.c_cpg.pkl", "rb") as f:
#     data = pickle.load(f)
# print(data.keys())


# import pickle
# import pandas as pd

# data = pickle.load(open("C:/Devign/devign/data/cpg/per_file/2.c_cpg.pkl", "rb"))
# print(type(data))
# print(data.head())
# print(type(data.iloc[0]["cpg"]))



# Count how many functions per original index
# import pandas as pd
# df = pd.read_pickle("data/cpg/1_cpg.pkl")
# print(df.groupby(level=0).size().head(10))

# # Inspect one index in detail
# print(df.loc[668].head(2))           # shows the first two function rows for index 668
# print(df.loc[668, "cpg"]["nodes"][:2])  # first two nodes from the first row




import argparse
import os
import sys
from typing import Any, Dict

import pandas as pd


def summarize_df(df: pd.DataFrame, limit: int = 10) -> None:
    print(f"rows: {len(df)} | columns: {list(df.columns)}")
    print("index sample:", list(df.index[:limit]))
    required = ["cpg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        return
    print(f"has func: {'func' in df.columns} | has target: {'target' in df.columns}")

    # show how many rows per original index (duplicate indices are normal)
    print("\ncounts per index (top 10):")
    print(df.groupby(level=0).size().head(10))


def safe_show_first_two_nodes_for_index(df: pd.DataFrame, idx: int) -> None:
    """
    Handles duplicate indices: df.loc[idx, 'cpg'] is a Series if there are multiple rows for that index.
    """
    if idx not in df.index:
        print(f"Index {idx} not found in DataFrame.")
        return

    # Extract the 'cpg' column for the given index
    cpg_sel = df.loc[idx, "cpg"]

    # Case 1: only one row for this index => cpg_sel is a dict
    if isinstance(cpg_sel, dict):
        nodes = cpg_sel.get("nodes", [])
        print(f"\nidx={idx} (single row) | nodes={len(nodes)}")
        print("first two nodes:", nodes[:2])
        return

    # Case 2: multiple rows for this index => cpg_sel is a Series
    if isinstance(cpg_sel, pd.Series):
        print(f"\nidx={idx} has {len(cpg_sel)} rows; iterating:")
        for pos, cpg in enumerate(cpg_sel.iloc[:2]):  # show first two rows for brevity
            if not isinstance(cpg, dict):
                print(f"  row#{pos}: cpg is not a dict (type={type(cpg)})")
                continue
            nodes = cpg.get("nodes", [])
            edges = cpg.get("edges", [])
            print(f"  row#{pos}: nodes={len(nodes)} edges={len(edges)}")
            print(f"  row#{pos}: first two nodes: {nodes[:2]}")
        return

    print(f"\nUnexpected type for df.loc[{idx}, 'cpg']: {type(cpg_sel)}")


def inspect_row(df: pd.DataFrame, idx: int, pos: int = 0) -> None:
    """
    Inspect a specific row among duplicates for a given index (by position).
    """
    if idx not in df.index:
        print(f"Index {idx} not found in DataFrame.")
        return

    rows = df.loc[idx]
    # rows can be a Series (single row) or DataFrame (multiple rows)
    if isinstance(rows, pd.Series):
        print(f"\nidx={idx} (single row):")
        print(rows.head())  # show some columns
        cpg = rows.get("cpg", {})
        print("nodes:", len(cpg.get("nodes", [])), "| edges:", len(cpg.get("edges", [])))
        return

    if isinstance(rows, pd.DataFrame):
        if pos >= len(rows):
            print(f"\nidx={idx} has {len(rows)} rows; pos={pos} is out of range.")
            return
        print(f"\nidx={idx} (row pos={pos}/{len(rows)}):")
        print(rows.iloc[pos].head())
        cpg = rows.iloc[pos]["cpg"]
        if isinstance(cpg, dict):
            print("nodes:", len(cpg.get("nodes", [])), "| edges:", len(cpg.get("edges", [])))
            print("first two nodes:", cpg.get("nodes", [])[:2])
        else:
            print("cpg is not a dict for this row.")
        return

    print(f"\nUnexpected type for df.loc[{idx}]: {type(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a *_cpg.pkl with duplicate-index safety.")
    parser.add_argument("--path", default=os.path.join("data", "cpg", "1_cpg.pkl"))
    parser.add_argument("--index", type=int, default=668, help="Index to inspect")
    parser.add_argument("--pos", type=int, default=0, help="Row position among duplicates to inspect")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: file not found: {args.path}")
        return 2

    df: pd.DataFrame = pd.read_pickle(args.path)

    summarize_df(df, limit=args.limit)
    safe_show_first_two_nodes_for_index(df, args.index)
    inspect_row(df, args.index, args.pos)

    print("\nok")
    return 0


if __name__ == "__main__":
    sys.exit(main())