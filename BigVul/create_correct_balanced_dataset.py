import json
import pandas as pd
import re
from collections import Counter

print("Creating balanced dataset from dataset.json...")

# Load the original dataset.json
with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Loaded {len(data)} records from dataset.json")

# Convert to DataFrame for easier processing
df = pd.DataFrame(data)

print("Original distribution:")
print(df['target'].value_counts())
print("\nProject distribution:")
print(df['project'].value_counts())

# Function to detect programming language from code
def detect_language(code):
    if pd.isna(code):
        return 'Unknown'
    
    code_str = str(code).lower()
    
    # C/C++ indicators
    if any(keyword in code_str for keyword in ['#include', 'malloc', 'free', 'printf', 'scanf', 'struct', 'typedef']):
        if any(keyword in code_str for keyword in ['cout', 'cin', 'std::', 'namespace', 'class', 'template']):
            return 'C++'
        return 'C'
    
    # Python indicators
    elif any(keyword in code_str for keyword in ['def ', 'import ', 'from ', 'print(', '__init__', 'self.']):
        return 'Python'
    
    # Java indicators  
    elif any(keyword in code_str for keyword in ['public class', 'private ', 'public static void main', 'import java']):
        return 'Java'
    
    # C# indicators
    elif any(keyword in code_str for keyword in ['using system', 'public class', 'namespace ', '.net']):
        return 'C#'
    
    # PHP indicators
    elif any(keyword in code_str for keyword in ['<?php', '$_', 'function ', 'echo ', '$this->']):
        return 'PHP'
    
    # Default to C for most cases (since dataset seems to be mostly C/C++)
    return 'C'

# Detect languages
print("Detecting programming languages...")
df['primary_language'] = df['func'].apply(detect_language)

print("Language distribution:")
print(df['primary_language'].value_counts())

# Filter for target languages
target_languages = ['C', 'C++', 'C#', 'Python', 'Java', 'PHP']
filtered_df = df[df['primary_language'].isin(target_languages)].copy()

print(f"\nFiltered dataset size: {len(filtered_df)}")
print("Filtered language distribution:")
print(filtered_df['primary_language'].value_counts())

# Function to extract filename from commit info (simplified)
def extract_filename(project, commit_id):
    # This is a simplified approach - in reality you'd need to query git
    # For now, we'll create a placeholder filename
    return f"{project.lower()}_commit_{commit_id[:8]}.c"

# Create the required fields
print("Creating required fields...")

# Add serial number
filtered_df = filtered_df.reset_index(drop=True)
filtered_df['snob'] = range(len(filtered_df))

# Map vulnerability field
filtered_df['vulnerability'] = filtered_df['target']

# Create placeholder fields (since original dataset.json doesn't have these)
filtered_df['cve_id'] = None  # Not available in dataset.json
filtered_df['severity'] = 'Unknown'  # Not available in dataset.json  
filtered_df['cwe_id'] = None  # Not available in dataset.json
filtered_df['filename_with_path'] = filtered_df.apply(lambda x: extract_filename(x['project'], x['commit_id']), axis=1)
filtered_df['line_number'] = None  # Not available in dataset.json
filtered_df['code_snippet'] = filtered_df['func']

# Select final columns in the required order
final_columns = [
    'snob', 'primary_language', 'vulnerability', 'cve_id', 
    'severity', 'cwe_id', 'filename_with_path', 'line_number', 'code_snippet'
]

result_df = filtered_df[final_columns].copy()

print(f"\nFinal dataset shape: {result_df.shape}")
print("Final language distribution:")
print(result_df['primary_language'].value_counts())
print("Final vulnerability distribution:")
print(result_df['vulnerability'].value_counts())

# Create balanced dataset for each language
balanced_datasets = {}

for lang in result_df['primary_language'].unique():
    lang_data = result_df[result_df['primary_language'] == lang].copy()
    
    if len(lang_data) == 0:
        continue
    
    vulnerable = lang_data[lang_data['vulnerability'] == 1]
    non_vulnerable = lang_data[lang_data['vulnerability'] == 0]
    
    print(f"\n{lang} - Vulnerable: {len(vulnerable)}, Non-Vulnerable: {len(non_vulnerable)}")
    
    if len(vulnerable) == 0 or len(non_vulnerable) == 0:
        print(f"Skipping {lang} - insufficient data for balancing")
        continue
    
    # Balance the dataset
    min_samples = min(len(vulnerable), len(non_vulnerable))
    max_samples_per_class = min(min_samples, 5000)  # Cap at 5000 per class
    
    if len(vulnerable) > max_samples_per_class:
        vulnerable_balanced = vulnerable.sample(n=max_samples_per_class, random_state=42)
    else:
        vulnerable_balanced = vulnerable
    
    if len(non_vulnerable) > max_samples_per_class:
        non_vulnerable_balanced = non_vulnerable.sample(n=max_samples_per_class, random_state=42)
    else:
        non_vulnerable_balanced = non_vulnerable
    
    balanced_lang_data = pd.concat([vulnerable_balanced, non_vulnerable_balanced], ignore_index=True)
    balanced_lang_data = balanced_lang_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    balanced_datasets[lang] = balanced_lang_data
    print(f"{lang} balanced dataset: {len(balanced_lang_data)} samples")

# Combine all balanced datasets
if balanced_datasets:
    final_balanced_df = pd.concat(balanced_datasets.values(), ignore_index=True)
    final_balanced_df['snob'] = range(len(final_balanced_df))  # Renumber
    
    print(f"\n=== FINAL BALANCED DATASET (FROM dataset.json) ===")
    print(f"Total samples: {len(final_balanced_df)}")
    print(f"\nLanguage distribution:")
    print(final_balanced_df['primary_language'].value_counts())
    print(f"\nVulnerability distribution:")
    print(final_balanced_df['vulnerability'].value_counts())
    
    # Save to CSV
    output_file = 'balanced_vulnerability_dataset_from_datasetjson.csv'
    final_balanced_df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Create JSON formats
    # Format 1: Regular JSON
    json_records = final_balanced_df.to_json(orient='records', indent=2)
    with open('balanced_vulnerability_dataset_from_datasetjson.json', 'w', encoding='utf-8') as f:
        f.write(json_records)
    
    # Format 2: Compact JSON
    json_compact = final_balanced_df.to_json(orient='records')
    with open('balanced_vulnerability_dataset_from_datasetjson_compact.json', 'w', encoding='utf-8') as f:
        f.write(json_compact)
    
    # Format 3: Structured JSON with metadata
    structured_data = {
        "metadata": {
            "source": "dataset.json",
            "total_samples": len(final_balanced_df),
            "languages": final_balanced_df['primary_language'].value_counts().to_dict(),
            "vulnerability_distribution": final_balanced_df['vulnerability'].value_counts().to_dict(),
            "fields": list(final_balanced_df.columns),
            "description": "Balanced vulnerability dataset from dataset.json",
            "format": "Each record contains vulnerability information with code snippets",
            "vulnerability_encoding": "0 = Non-Vulnerable, 1 = Vulnerable",
            "note": "CVE ID, CWE ID, and severity fields are not available in original dataset.json"
        },
        "data": final_balanced_df.to_dict('records')
    }
    
    with open('balanced_vulnerability_dataset_from_datasetjson_structured.json', 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2)
    
    print("✅ All JSON formats created:")
    print("  - balanced_vulnerability_dataset_from_datasetjson.json")
    print("  - balanced_vulnerability_dataset_from_datasetjson_compact.json") 
    print("  - balanced_vulnerability_dataset_from_datasetjson_structured.json")
    
    # Display sample data
    print(f"\n=== SAMPLE DATA ===")
    print(final_balanced_df.head())
    
    # Show statistics per language
    print(f"\n=== PER-LANGUAGE STATISTICS ===")
    for lang in final_balanced_df['primary_language'].unique():
        lang_subset = final_balanced_df[final_balanced_df['primary_language'] == lang]
        vuln_count = len(lang_subset[lang_subset['vulnerability'] == 1])
        non_vuln_count = len(lang_subset[lang_subset['vulnerability'] == 0])
        print(f"{lang}: {len(lang_subset)} total ({vuln_count} vulnerable, {non_vuln_count} non-vulnerable)")

else:
    print("No balanced datasets could be created.")