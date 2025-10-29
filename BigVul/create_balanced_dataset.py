from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.utils import resample
import re

print("Loading BigVul dataset from Hugging Face...")
dataset = load_dataset("bstee615/bigvul", "default")

# Convert all splits to pandas DataFrames
train_df = dataset['train'].to_pandas()
val_df = dataset['validation'].to_pandas()
test_df = dataset['test'].to_pandas()

# Combine all splits for comprehensive analysis
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Total dataset size: {len(full_df)}")
print(f"Available languages: {full_df['lang'].value_counts()}")

# Define target languages
target_languages = ['C', 'C++', 'C#', 'Python', 'Java', 'PHP']

# Filter for target languages (case-insensitive matching)
def normalize_language(lang):
    if pd.isna(lang):
        return None
    lang = str(lang).strip().lower()
    if lang in ['c']:
        return 'C'
    elif lang in ['c++', 'cpp']:
        return 'C++'
    elif lang in ['c#', 'csharp']:
        return 'C#'
    elif lang in ['python', 'py']:
        return 'Python'
    elif lang in ['java']:
        return 'Java'
    elif lang in ['php']:
        return 'PHP'
    return None

# Apply language normalization
full_df['normalized_lang'] = full_df['lang'].apply(normalize_language)

# Filter for target languages
filtered_df = full_df[full_df['normalized_lang'].isin(target_languages)].copy()

print(f"\nFiltered dataset size: {len(filtered_df)}")
print(f"Language distribution after filtering:")
print(filtered_df['normalized_lang'].value_counts())

print(f"\nVulnerability distribution:")
print(filtered_df['vul'].value_counts())

# Function to extract file name and line number from codeLink
def extract_file_info(code_link):
    if pd.isna(code_link):
        return None, None
    
    # Try to extract filename from URL
    filename = None
    line_number = None
    
    # Look for filename pattern in URL
    if isinstance(code_link, str):
        # Extract filename (last part after /)
        parts = code_link.split('/')
        if len(parts) > 0:
            filename = parts[-1]
            # Remove any query parameters or anchors
            if '#' in filename:
                filename = filename.split('#')[0]
            if '?' in filename:
                filename = filename.split('?')[0]
        
        # Try to extract line number from URL (common patterns)
        line_match = re.search(r'#L(\d+)', code_link)
        if line_match:
            line_number = int(line_match.group(1))
        else:
            # Look for line parameter
            line_match = re.search(r'line=(\d+)', code_link)
            if line_match:
                line_number = int(line_match.group(1))
    
    return filename, line_number

# Extract file info
print("\nExtracting file information...")
file_info = filtered_df['codeLink'].apply(extract_file_info)
filtered_df['filename'] = [info[0] for info in file_info]
filtered_df['line_number'] = [info[1] for info in file_info]

# Function to extract code snippet (using func_before for vulnerable, func_after for context)
def get_code_snippet(row):
    if pd.notna(row['func_before']) and pd.notna(row['func_after']):
        if row['vul'] == 1:
            return row['func_before']  # Vulnerable code
        else:
            return row['func_after']   # Non-vulnerable code
    elif pd.notna(row['func_before']):
        return row['func_before']
    elif pd.notna(row['func_after']):
        return row['func_after']
    return None

filtered_df['code_snippet'] = filtered_df.apply(get_code_snippet, axis=1)

# Create severity mapping (this is a simplified mapping as BigVul doesn't have explicit severity)
def map_severity(cwe_id):
    if pd.isna(cwe_id):
        return 'Unknown'
    
    # High severity CWEs (buffer overflows, code execution, etc.)
    high_severity = ['CWE-119', 'CWE-120', 'CWE-121', 'CWE-122', 'CWE-124', 'CWE-125', 
                     'CWE-787', 'CWE-788', 'CWE-416', 'CWE-415', 'CWE-190', 'CWE-191']
    
    # Medium severity CWEs (information disclosure, DoS, etc.)
    medium_severity = ['CWE-200', 'CWE-399', 'CWE-362', 'CWE-369', 'CWE-476', 'CWE-401']
    
    # Low severity CWEs (input validation, etc.)
    low_severity = ['CWE-20', 'CWE-189', 'CWE-264', 'CWE-79', 'CWE-89']
    
    cwe_str = str(cwe_id)
    if any(cwe in cwe_str for cwe in high_severity):
        return 'High'
    elif any(cwe in cwe_str for cwe in medium_severity):
        return 'Medium'
    elif any(cwe in cwe_str for cwe in low_severity):
        return 'Low'
    else:
        return 'Medium'  # Default to medium for unknown CWEs

filtered_df['severity'] = filtered_df['CWE ID'].apply(map_severity)

# Create the final dataset with required fields
final_columns = {
    'snob': range(len(filtered_df)),  # Serial number
    'primary_language': filtered_df['normalized_lang'],
    'vulnerability': filtered_df['vul'].map({0: 'Non-Vulnerable', 1: 'Vulnerable'}),
    'cve_id': filtered_df['CVE ID'],
    'severity': filtered_df['severity'],
    'cwe_id': filtered_df['CWE ID'],
    'filename_with_path': filtered_df['filename'],
    'line_number': filtered_df['line_number'],
    'code_snippet': filtered_df['code_snippet']
}

result_df = pd.DataFrame(final_columns)

print(f"\nFinal dataset shape: {result_df.shape}")
print(f"\nLanguage distribution in final dataset:")
print(result_df['primary_language'].value_counts())
print(f"\nVulnerability distribution in final dataset:")
print(result_df['vulnerability'].value_counts())

# Create balanced dataset for each language
balanced_datasets = {}

for lang in target_languages:
    lang_data = result_df[result_df['primary_language'] == lang].copy()
    
    if len(lang_data) == 0:
        print(f"\nNo data found for {lang}")
        continue
    
    vulnerable = lang_data[lang_data['vulnerability'] == 'Vulnerable']
    non_vulnerable = lang_data[lang_data['vulnerability'] == 'Non-Vulnerable']
    
    print(f"\n{lang} - Vulnerable: {len(vulnerable)}, Non-Vulnerable: {len(non_vulnerable)}")
    
    if len(vulnerable) == 0 or len(non_vulnerable) == 0:
        print(f"Skipping {lang} - insufficient data for balancing")
        continue
    
    # Balance the dataset
    min_samples = min(len(vulnerable), len(non_vulnerable))
    max_samples_per_class = min(min_samples, 5000)  # Cap at 5000 per class for manageability
    
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
    
    print(f"\n=== FINAL BALANCED DATASET ===")
    print(f"Total samples: {len(final_balanced_df)}")
    print(f"\nLanguage distribution:")
    print(final_balanced_df['primary_language'].value_counts())
    print(f"\nVulnerability distribution:")
    print(final_balanced_df['vulnerability'].value_counts())
    print(f"\nSeverity distribution:")
    print(final_balanced_df['severity'].value_counts())
    
    # Save to CSV
    output_file = 'balanced_vulnerability_dataset.csv'
    final_balanced_df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Display sample data
    print(f"\n=== SAMPLE DATA ===")
    print(final_balanced_df.head(10))
    
    # Show statistics per language
    print(f"\n=== PER-LANGUAGE STATISTICS ===")
    for lang in final_balanced_df['primary_language'].unique():
        lang_subset = final_balanced_df[final_balanced_df['primary_language'] == lang]
        vuln_count = len(lang_subset[lang_subset['vulnerability'] == 'Vulnerable'])
        non_vuln_count = len(lang_subset[lang_subset['vulnerability'] == 'Non-Vulnerable'])
        print(f"{lang}: {len(lang_subset)} total ({vuln_count} vulnerable, {non_vuln_count} non-vulnerable)")

else:
    print("No balanced datasets could be created.")