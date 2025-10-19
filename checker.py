import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd


def load_configs(config_path: str = "configs.json") -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reproduce_select_filter(dataset: pd.DataFrame) -> pd.DataFrame:
    # Mirrors select() in main.py: project == "FFmpeg" and len(func) < 1200
    df = dataset.loc[dataset['project'] == "FFmpeg"].copy()
    len_filter = df.func.str.len() < 1200
    df = df.loc[len_filter]
    return df


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    # Mirrors data.clean() and drop of commit_id, project
    df = df.drop_duplicates(subset="func", keep=False)
    for col in ("commit_id", "project"):
        if col in df.columns:
            del df[col]
    return df


def slice_frame(df: pd.DataFrame, size: int) -> List[Tuple[int, pd.DataFrame]]:
    # Mirrors data.slice_frame() then the list comp used in main.py
    if size <= 0:
        return [(0, df)]
    grouped = df.groupby((pd.RangeIndex(len(df)) // size))
    return [(s, grp.apply(lambda x: x)) for s, grp in grouped]


def expect_paths(paths_cfg: Dict, files_cfg: Dict) -> Dict[str, str]:
    return {
        "raw_json": os.path.join(paths_cfg["raw"], files_cfg["raw"]),
        "cpg_dir": paths_cfg["cpg"],
        "joern_tmp": paths_cfg.get("joern", "data/joern/"),
    }


def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _json_indices_from_file_field(functions: List[Dict]) -> set:
    idxs = set()
    for fobj in functions:
        try:
            file_name = fobj.get("file", "")
            base = os.path.basename(file_name)
            if base.endswith(".c"):
                idx = int(base.split(".c")[0])
                idxs.add(idx)
        except Exception:
            # ignore unparsable entries
            pass
    return idxs


def validate_json_structure(json_path: str) -> Tuple[bool, str, int, set]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Accepts both top-level {"functions": [...]} or empty lists
        functions = obj.get("functions", [])
        if not isinstance(functions, list):
            return False, "'functions' is not a list", 0, set()
        # Light structural checks on a few entries
        ok_count = 0
        for fobj in functions[:10]:
            if not isinstance(fobj, dict):
                return False, "function entry is not a dict", ok_count, set()
            # Allow missing optional keys, but prefer standard ones
            _nodes = fobj.get("nodes")
            _edges = fobj.get("edges")
            if _nodes is None or _edges is None:
                # Some scripts may wrap differently; allow but flag
                pass
            ok_count += 1
        idxs = _json_indices_from_file_field(functions)
        return True, "ok", len(functions), idxs
    except Exception as e:
        return False, f"json error: {e}", 0, set()


def _normalize_func_to_str(val) -> str:
    if isinstance(val, dict):
        return val.get("function", "")
    return str(val)


def validate_pkl_slice(pkl_path: str, expected_indices: set, *, strict: bool = False, filtered_slice: pd.DataFrame = None) -> Tuple[bool, str, int]:
    try:
        df: pd.DataFrame = pd.read_pickle(pkl_path)
        if df.empty:
            return False, "empty dataframe", 0

        # Required columns
        if "cpg" not in df.columns:
            return False, "missing column 'cpg'", len(df)

        # Index alignment: pkl index should be subset of the expected slice indices
        idx = set(df.index.tolist())
        if not idx:
            return False, "no indices in pkl", 0
        if strict:
            if idx != expected_indices:
                return False, "indices not equal to expected slice", len(df)
        else:
            if not (idx.issubset(expected_indices)):
                return False, "indices not subset of expected slice", len(df)

        # Spot-check a few cpg entries
        sample = df["cpg"].head(5).tolist()
        for entry in sample:
            if not isinstance(entry, dict):
                return False, "cpg entry is not a dict", len(df)
            # Accepts two styles: {"functions": [...]} OR {"nodes": [...], "edges": [...]}
            if ("functions" not in entry) and ("nodes" not in entry):
                return False, "cpg dict missing 'functions' and 'nodes'", len(df)

        # Strict checks: func text equality against filtered slice
        if strict and filtered_slice is not None and "func" in df.columns and "func" in filtered_slice.columns:
            # Align on indices
            sub = df[["func"]].join(filtered_slice[["func"]], how="left", rsuffix="_orig")
            # normalize both sides to strings
            left = sub["func"].map(_normalize_func_to_str)
            right = sub["func_orig"].map(_normalize_func_to_str)
            mism = (left != right)
            if mism.any():
                bad = sub[mism].index.tolist()[:5]
                return False, f"func mismatch at indices: {bad}", len(df)

        return True, "ok", len(df)
    except Exception as e:
        return False, f"pkl error: {e}", 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate -c (Create) pipeline outputs")
    parser.add_argument("--config", default="configs.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Enable strict 1:1 validation against dataset.json")
    args = parser.parse_args()

    cfg = load_configs(args.config)
    paths = expect_paths(cfg["paths"], cfg["files"])

    raw_json_path = paths["raw_json"]
    cpg_dir = paths["cpg_dir"]

    errors: List[str] = []
    warnings: List[str] = []

    # 1) Load and filter raw dataset like main.py
    if not file_exists(raw_json_path):
        errors.append(f"Missing raw dataset: {raw_json_path}")
        print("FAIL")
        print("\n".join(errors))
        return 2

    raw_df = pd.read_json(raw_json_path)
    filtered = reproduce_select_filter(raw_df)
    filtered = clean_and_prepare(filtered)

    slice_size = int(cfg["create"]["slice_size"]) if cfg.get("create") else 100
    slices = slice_frame(filtered, slice_size)

    if not file_exists(cpg_dir):
        errors.append(f"Missing cpg dir: {cpg_dir}")
        print("FAIL")
        print("\n".join(errors))
        return 2

    # 2) For each slice s, check .bin, .json, .pkl
    all_ok = True
    for s, slice_df in slices:
        expected_idx = set(slice_df.index.tolist())
        bin_path = os.path.join(cpg_dir, f"{s}_cpg.bin")
        json_path = os.path.join(cpg_dir, f"{s}_cpg.json")
        pkl_path = os.path.join(cpg_dir, f"{s}_cpg.pkl")

        # .bin presence
        if not file_exists(bin_path):
            all_ok = False
            errors.append(f"[s={s}] missing bin: {bin_path}")

        # .json presence + structure
        if not file_exists(json_path):
            all_ok = False
            errors.append(f"[s={s}] missing json: {json_path}")
        else:
            ok, msg, nfunc, json_idxs = validate_json_structure(json_path)
            if not ok:
                all_ok = False
                errors.append(f"[s={s}] invalid json: {json_path} ({msg})")
            elif args.verbose:
                print(f"[s={s}] json ok ({nfunc} functions)")

            # Strict: ensure JSON file set covers expected indices
            if args.strict:
                if not expected_idx.issubset(json_idxs):
                    all_ok = False
                    missing = sorted(list(expected_idx - json_idxs))[:10]
                    errors.append(f"[s={s}] json missing file indices from slice: {missing}")

        # .pkl presence + content
        if not file_exists(pkl_path):
            all_ok = False
            errors.append(f"[s={s}] missing pkl: {pkl_path}")
        else:
            ok, msg, nrows = validate_pkl_slice(pkl_path, expected_idx, strict=args.strict, filtered_slice=slice_df)
            if not ok:
                all_ok = False
                errors.append(f"[s={s}] invalid pkl: {pkl_path} ({msg})")
            elif args.verbose:
                print(f"[s={s}] pkl ok ({nrows} rows)")

    # Summary
    if all_ok:
        print("PASS")
        if args.verbose:
            print(f"Checked {len(slices)} slices successfully.")
        return 0

    print("FAIL")
    # De-duplicate errors for readability
    seen = set()
    for e in errors:
        if e in seen:
            continue
        seen.add(e)
        print(e)
    return 1


if __name__ == "__main__":
    sys.exit(main())


