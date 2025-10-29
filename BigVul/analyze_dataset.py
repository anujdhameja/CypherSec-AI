import pandas as pd

# Load the balanced dataset
df = pd.read_csv('balanced_vulnerability_dataset.csv')

print("=== BALANCED VULNERABILITY DATASET ANALYSIS ===")
print(f"Total samples: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n=== COLUMN DETAILS ===")
for i, col in enumerate(df.columns, 1):
    print(f"{chr(96+i)}. {col}")

print("\n=== DATASET OVERVIEW ===")
print(df.info())

print("\n=== LANGUAGE DISTRIBUTION ===")
lang_dist = df['primary_language'].value_counts()
for lang, count in lang_dist.items():
    vuln_count = len(df[(df['primary_language'] == lang) & (df['vulnerability'] == 'Vulnerable')])
    non_vuln_count = len(df[(df['primary_language'] == lang) & (df['vulnerability'] == 'Non-Vulnerable')])
    print(f"{lang}: {count} total ({vuln_count} vulnerable, {non_vuln_count} non-vulnerable)")

print("\n=== VULNERABILITY DISTRIBUTION ===")
print(df['vulnerability'].value_counts())

print("\n=== SEVERITY DISTRIBUTION ===")
print(df['severity'].value_counts())

print("\n=== TOP 10 CWE IDs ===")
cwe_counts = df['cwe_id'].value_counts().head(10)
print(cwe_counts)

print("\n=== CVE ID STATISTICS ===")
total_cves = df['cve_id'].notna().sum()
unique_cves = df['cve_id'].nunique()
print(f"Total samples with CVE ID: {total_cves}")
print(f"Unique CVE IDs: {unique_cves}")

print("\n=== FILE INFORMATION STATISTICS ===")
files_with_name = df['filename_with_path'].notna().sum()
files_with_line = df['line_number'].notna().sum()
print(f"Samples with filename: {files_with_name}")
print(f"Samples with line number: {files_with_line}")

print("\n=== CODE SNIPPET STATISTICS ===")
snippets_available = df['code_snippet'].notna().sum()
avg_snippet_length = df['code_snippet'].str.len().mean()
print(f"Samples with code snippets: {snippets_available}")
print(f"Average code snippet length: {avg_snippet_length:.0f} characters")

print("\n=== SAMPLE RECORDS ===")
print("Vulnerable sample:")
vuln_sample = df[df['vulnerability'] == 'Vulnerable'].iloc[0]
for col in df.columns:
    value = vuln_sample[col]
    if col == 'code_snippet' and pd.notna(value):
        # Show first 200 characters of code snippet
        value = str(value)[:200] + "..." if len(str(value)) > 200 else value
    print(f"  {col}: {value}")

print("\nNon-vulnerable sample:")
non_vuln_sample = df[df['vulnerability'] == 'Non-Vulnerable'].iloc[0]
for col in df.columns:
    value = non_vuln_sample[col]
    if col == 'code_snippet' and pd.notna(value):
        # Show first 200 characters of code snippet
        value = str(value)[:200] + "..." if len(str(value)) > 200 else value
    print(f"  {col}: {value}")

print(f"\n=== DATASET SAVED AS ===")
print("File: balanced_vulnerability_dataset.csv")
print("Format: CSV with headers")
print("Ready for machine learning and analysis!")