import json
import os

def analyze_json_file(file_path):
    print(f"Analyzing file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    print(f"File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    
    # Try to read first 2KB of the file
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        first_chunk = f.read(2048)
        print("\nFirst 200 characters:")
        print("-" * 50)
        print(first_chunk[:200])
        print("-" * 50)
        
        # Try to find the first newline to see if it's a JSON Lines file
        first_newline = first_chunk.find('\n')
        if first_newline > 0:
            first_line = first_chunk[:first_newline].strip()
            print("\nFirst line (potential JSON object):")
            print("-" * 50)
            print(first_line)
            print("-" * 50)
            
            try:
                # Try to parse the first line as JSON
                first_obj = json.loads(first_line)
                print("\nSuccessfully parsed first line as JSON!")
                print(f"Type: {type(first_obj)}")
                print("Keys:", list(first_obj.keys()) if isinstance(first_obj, dict) else "N/A")
                return
            except json.JSONDecodeError as e:
                print(f"\nFailed to parse first line as JSON: {e}")
    
    # If we get here, try reading the whole file as JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print("\nSuccessfully parsed entire file as JSON!")
            print(f"Type: {type(data)}")
            if isinstance(data, list):
                print(f"List length: {len(data)}")
                if data:
                    print("First item type:", type(data[0]))
            elif isinstance(data, dict):
                print("Keys:", list(data.keys()))
    except json.JSONDecodeError as e:
        print(f"\nFailed to parse file as JSON: {e}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")

if __name__ == "__main__":
    file_path = r'C:\Devign\devign\data\raw\dataset.json'
    analyze_json_file(file_path)
