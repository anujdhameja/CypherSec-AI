import argparse
import os
import sys
from typing import Any, Dict

import pandas as pd


def inspect_cpg(cpg_obj: Dict[str, Any]) -> str:
    if not isinstance(cpg_obj, dict):
        return "cpg is not a dict"
    nodes = cpg_obj.get("nodes", [])
    edges = cpg_obj.get("edges", [])
    parts = [
        f"nodes: {len(nodes)}",
        f"edges: {len(edges)}",
    ]
    # show a tiny sample
    if nodes:
        n0 = nodes[0]
        parts.append(
            "sample_node: {id} | {label} | code={code}".format(
                id=n0.get("id"), label=n0.get("label"), code=(n0.get("code") or "")[:60]
            )
        )
    if edges:
        e0 = edges[0]
        parts.append(
            "sample_edge: {src}->{tgt} ({lbl})".format(
                src=e0.get("source") or e0.get("in"),
                tgt=e0.get("target") or e0.get("out"),
                lbl=e0.get("label"),
            )
        )
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and verify structure of a *_cpg.pkl file")
    parser.add_argument("--path", default=os.path.join("data", "cpg", "1_cpg.pkl"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"ERROR: File not found: {args.path}")
        return 2

    df: pd.DataFrame = pd.read_pickle(args.path)

    print(f"loaded: {args.path}")
    print(f"rows: {len(df)} | columns: {list(df.columns)}")
    print("index sample:", list(df.index[: args.limit]))

    # basic schema checks
    required = ["cpg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: missing columns: {missing}")
        return 1

    # optional columns
    has_func = "func" in df.columns
    has_target = "target" in df.columns
    print(f"has func: {has_func} | has target: {has_target}")

    # inspect a few rows
    sample = df.head(args.limit)
    for idx, row in sample.iterrows():
        cpg_obj = row["cpg"]
        print(f"idx={idx} | {inspect_cpg(cpg_obj)}")

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())


