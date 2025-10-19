import os
import pickle

def check_cpg_file(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
            
        print(f"Checking file: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        # Try to read the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Print basic information about the data
        print(f"\nType of data: {type(data)}")
        
        # If it's a pandas DataFrame
        if hasattr(data, 'shape'):
            print(f"DataFrame shape: {data.shape}")
            print("\nColumns:", list(data.columns))
            print("\nFirst few rows:")
            print(data.head())
        # If it's a dictionary
        elif isinstance(data, dict):
            print(f"Dictionary with {len(data)} keys")
            print("Keys:", list(data.keys()))
            
            # Print first few items if it's not too large
            for i, (k, v) in enumerate(data.items()):
                if i >= 3:  # Only show first 3 items
                    print("...")
                    break
                print(f"\nKey: {k}")
                print(f"Value type: {type(v)}")
                if hasattr(v, '__len__'):
                    print(f"Length: {len(v)}")
                print(f"Sample: {str(v)[:200]}...")
                
        # For other types
        else:
            print(f"Data content (first 500 chars):"
            print(str(data)[:500])
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_cpg_file('data/cpg/0_cpg.pkl')
