from datasets import load_dataset

print("Loading BigVul dataset from Hugging Face...")
# This will download and load the dataset. It might take a few minutes.
# The 'default' config is the 180k balanced dataset.
dataset = load_dataset("bstee615/bigvul", "default")

print("Dataset loaded successfully!")
print(dataset)



#analyse


import pandas as pd

# Convert the 'train' split to a pandas DataFrame
print("Converting to pandas DataFrame...")
df = dataset['train'].to_pandas()

# 1. Analyze the Columns and Data Types
print("\n--- Dataset Info ---")
df.info()

# 2. See the first few rows to understand the data
print("\n--- Dataset Head ---")
print(df.head())

# 3. Analyze the 'vul' column (Vulnerable vs. Non-Vulnerable)
print("\n--- Vulnerability Distribution (vul column) ---")
# The 'vul' column is 1 for vulnerable, 0 for non-vulnerable
print(df['vul'].value_counts())

# 4. Analyze the 'CWE ID' column for vulnerable samples
print("\n--- Top 10 CWEs for Vulnerable Samples ---")
# Filter for only vulnerable samples
vulnerable_df = df[df['vul'] == 1]
print(vulnerable_df['CWE ID'].value_counts().head(10))