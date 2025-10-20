import os
import sys
import torch
import pandas as pd
from torch_geometric.data import Data

print("="*70)
print("DEBUGGING DATA LOADING")
print("="*70)

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Check if data directory exists
data_dir = os.path.join(project_root, 'data', 'input')
print(f"Data directory: {data_dir}")
print(f"Directory exists: {os.path.exists(data_dir)}")

# List files in the directory
try:
    files = os.listdir(data_dir)
    print(f"\nFound {len(files)} files in data directory")
    print("First 5 files:", files[:5])
    
    # Try loading the first file
    if files:
        first_file = os.path.join(data_dir, files[0])
        print(f"\nLoading first file: {first_file}")
        
        try:
            # Try loading with pandas
            data = pd.read_pickle(first_file)
            print("\nFile loaded successfully!")
            print(f"Type: {type(data)}")
            
            # Print some basic info about the data
            if hasattr(data, 'shape'):
                print(f"Data shape: {data.shape}")
            if hasattr(data, 'columns'):
                print(f"Columns: {data.columns.tolist()}")
            if hasattr(data, 'head'):
                print("\nFirst few rows:")
                print(data.head())
            
            # If it's a DataFrame with 'input' column
            if hasattr(data, 'input'):
                print("\nInput column type:", type(data.input))
                print("First input item type:", type(data.input.iloc[0]) if hasattr(data.input, 'iloc') else 'N/A')
            
            # If it's a list of PyG Data objects
            elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Data):
                print(f"\nFound {len(data)} PyG Data objects")
                print("First Data object keys:", data[0].keys)
            
        except Exception as e:
            print(f"\nError loading file: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"\nError listing directory: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DEBUGGING COMPLETE")
print("="*70)
