#!/usr/bin/env python3
"""
Fix dataset validation issues
"""

import json
from pathlib import Path

def fix_dataset_issues():
    """Fix common issues in datasets"""
    
    base_path = Path("C:/Devign/devign/Custom datasets/data")
    
    # Fix individual datasets
    datasets = [
        "dataset_devign_mapped.json",
        "dataset_juliet_mapped.json", 
        "dataset_github_mapped.json",
        "dataset_owasp_mapped.json"
    ]
    
    for dataset_file in datasets:
        dataset_path = base_path / dataset_file
        if not dataset_path.exists():
            continue
            
        print(f"Fixing {dataset_file}...")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fixed_count = 0
        for entry in data:
            # Fix missing is_vulnerable field
            if "is_vulnerable" not in entry:
                # Infer from other fields
                if "safe" in entry.get("function", "").lower() or "good" in entry.get("file", "").lower():
                    entry["is_vulnerable"] = False
                else:
                    entry["is_vulnerable"] = True
                fixed_count += 1
            
            # Ensure all required fields exist
            if "cwe_id" not in entry:
                entry["cwe_id"] = "CWE-Unknown"
                fixed_count += 1
            
            if "severity" not in entry:
                if entry.get("is_vulnerable", False):
                    entry["severity"] = "MEDIUM"
                else:
                    entry["severity"] = "LOW"
                fixed_count += 1
            
            if "cve_ids" not in entry:
                entry["cve_ids"] = []
                fixed_count += 1
        
        # Save fixed dataset
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Fixed {fixed_count} issues in {dataset_file}")
    
    print("Regenerating combined dataset...")
    
    # Reload and recombine all datasets
    all_data = []
    for dataset_file in datasets:
        dataset_path = base_path / dataset_file
        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
    
    # Save fixed combined dataset
    combined_path = base_path / "dataset_combined.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Updated combined dataset with {len(all_data)} entries")

if __name__ == "__main__":
    fix_dataset_issues()