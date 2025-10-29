import pandas as pd
import json

print("Converting balanced vulnerability dataset from CSV to JSON...")

# Load the CSV dataset
df = pd.read_csv('balanced_vulnerability_dataset.csv')

print(f"Loaded dataset with {len(df)} samples")

# Convert to JSON format
# Option 1: Records format (array of objects)
json_records = df.to_json(orient='records', indent=2)

# Save as JSON file
output_file = 'balanced_vulnerability_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(json_records)

print(f"Dataset converted and saved as: {output_file}")

# Verify the conversion
with open(output_file, 'r', encoding='utf-8') as f:
    loaded_data = json.load(f)

print(f"Verification: JSON file contains {len(loaded_data)} records")
print(f"Sample record structure:")
if loaded_data:
    sample_record = loaded_data[0]
    for key, value in sample_record.items():
        # Truncate long code snippets for display
        if key == 'code_snippet' and value and len(str(value)) > 100:
            display_value = str(value)[:100] + "..."
        else:
            display_value = value
        print(f"  {key}: {display_value}")

print(f"\nJSON conversion completed successfully!")
print(f"File size comparison:")
import os
csv_size = os.path.getsize('balanced_vulnerability_dataset.csv')
json_size = os.path.getsize(output_file)
print(f"CSV: {csv_size:,} bytes")
print(f"JSON: {json_size:,} bytes")