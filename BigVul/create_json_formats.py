import pandas as pd
import json

print("Creating multiple JSON format options...")

# Load the CSV dataset
df = pd.read_csv('balanced_vulnerability_dataset.csv')

# Format 1: Records format (array of objects) - Already created
print("✅ Format 1: Records format already created as balanced_vulnerability_dataset.json")

# Format 2: Compact format (no indentation)
print("Creating Format 2: Compact JSON...")
json_compact = df.to_json(orient='records')
with open('balanced_vulnerability_dataset_compact.json', 'w', encoding='utf-8') as f:
    f.write(json_compact)

# Format 3: Structured format with metadata
print("Creating Format 3: Structured JSON with metadata...")
structured_data = {
    "metadata": {
        "total_samples": len(df),
        "languages": df['primary_language'].value_counts().to_dict(),
        "vulnerability_distribution": df['vulnerability'].value_counts().to_dict(),
        "severity_distribution": df['severity'].value_counts().to_dict(),
        "top_cwe_ids": df['cwe_id'].value_counts().head(5).to_dict(),
        "fields": list(df.columns),
        "description": "Balanced vulnerability dataset from BigVul",
        "format": "Each record contains vulnerability information with code snippets"
    },
    "data": df.to_dict('records')
}

with open('balanced_vulnerability_dataset_structured.json', 'w', encoding='utf-8') as f:
    json.dump(structured_data, f, indent=2)

# Show file sizes
import os
files = [
    'balanced_vulnerability_dataset.json',
    'balanced_vulnerability_dataset_compact.json', 
    'balanced_vulnerability_dataset_structured.json'
]

print("\n=== JSON FORMAT OPTIONS ===")
for i, file in enumerate(files, 1):
    size = os.path.getsize(file)
    print(f"Format {i}: {file}")
    print(f"  Size: {size:,} bytes ({size/1024/1024:.1f} MB)")
    
    # Show structure sample
    with open(file, 'r', encoding='utf-8') as f:
        if file.endswith('_structured.json'):
            data = json.load(f)
            print(f"  Structure: metadata + {len(data['data'])} records")
            print(f"  Sample keys: {list(data.keys())}")
        else:
            content = f.read(200)
            print(f"  Sample: {content}...")
    print()

print("All JSON formats created successfully!")
print("\nRecommended usage:")
print("- Use 'compact' for minimal file size")
print("- Use 'structured' for applications needing metadata")
print("- Use regular format for general purpose")