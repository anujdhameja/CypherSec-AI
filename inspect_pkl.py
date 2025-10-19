import os
import pickle
import pandas as pd
from pprint import pprint

def inspect_pkl_file(filepath):
    print(f"\nInspecting: {filepath}")
    print("=" * 80)
    
    try:
        # Try to load the pickle file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print("\nData type:", type(data))
        
        # If it's a DataFrame, show columns and first few rows
        if isinstance(data, pd.DataFrame):
            print("\nDataFrame Info:")
            print("-" * 40)
            print(f"Shape: {data.shape}")
            print("\nColumns:", list(data.columns))
            
            # Show first few rows
            print("\nFirst 2 rows:")
            for i, row in data.head(2).iterrows():
                print(f"\nRow {i}:")
                for col in data.columns:
                    val = row[col]
                    print(f"  {col}: {type(val).__name__}")
                    if isinstance(val, (dict, list)):
                        print(f"    Length: {len(val) if hasattr(val, '__len__') else 'N/A'}")
                        if isinstance(val, dict):
                            print("    Keys:", list(val.keys())[:5], "..." if len(val) > 5 else "")
                    elif hasattr(val, '__dict__'):
                        print("    Attributes:", list(vars(val).keys())[:5], "..." if len(vars(val)) > 5 else "")
        
        # If it's a dictionary, show its structure
        elif isinstance(data, dict):
            print("\nDictionary structure:")
            print("-" * 40)
            print("Top-level keys:", list(data.keys()))
            
            # Show first few items for each key
            for key, value in list(data.items())[:3]:  # Only show first 3 items
                print(f"\nKey: {key} ({type(value).__name__})")
                if isinstance(value, (list, dict, set)):
                    print(f"  Length: {len(value)}")
                    if value:
                        sample = list(value)[0] if hasattr(value, '__iter__') and not isinstance(value, dict) else None
                        if sample is not None:
                            print(f"  Sample item type: {type(sample).__name__}")
                elif hasattr(value, '__dict__'):
                    print("  Attributes:", list(vars(value).keys())[:5], "..." if len(vars(value)) > 5 else "")
        
        # For other types, show their string representation
        else:
            print("\nContent:")
            print("-" * 40)
            pprint(data)
            
    except Exception as e:
        print(f"\nError loading file: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inspect_pkl.py <path_to_pkl_file>")
        return
    
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    inspect_pkl_file(filepath)

if __name__ == "__main__":
    main()
