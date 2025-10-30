#!/usr/bin/env python3
"""
Create separate language-specific JSON files from the multi-language dataset
"""

import json
import pandas as pd

def create_language_datasets():
    """Create separate JSON files for each language"""
    
    # Load the multi-language labeled dataset
    with open('data/raw/dataset_tester_multilang_labeled.json', 'r') as f:
        dataset = json.load(f)
    
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} samples with languages: {df['language'].unique()}")
    
    # Group by language and create separate files
    language_groups = df.groupby('language')
    
    created_files = []
    
    for lang, group in language_groups:
        # Create filename
        filename = f"data/raw/{lang}_dataset.json"
        
        # Convert to list of dictionaries
        lang_data = group.to_dict('records')
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(lang_data, f, indent=2)
        
        created_files.append(filename)
        
        print(f"✅ Created {filename}: {len(lang_data)} samples")
        
        # Show sample
        if len(lang_data) > 0:
            sample = lang_data[0]
            func_preview = sample['func'][:50].replace('\n', ' ')
            target_label = "VULNERABLE" if sample['target'] == 1 else "SAFE"
            print(f"   Sample: [{target_label}] {func_preview}...")
    
    print(f"\n📊 Summary:")
    print(f"  Total files created: {len(created_files)}")
    print(f"  Files: {[f.split('/')[-1] for f in created_files]}")
    
    return created_files

if __name__ == "__main__":
    create_language_datasets()