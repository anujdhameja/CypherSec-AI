#!/usr/bin/env python3
"""
Dataset Quality Validation Script
"""

import json
from pathlib import Path
from typing import Dict, List, Any

def validate_dataset_quality(dataset_path: str) -> Dict[str, Any]:
    """Validate dataset quality and completeness"""
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    issues = []
    stats = {
        "total_entries": len(data),
        "valid_entries": 0,
        "issues_found": 0
    }
    
    required_fields = ["id", "source", "file", "function", "code", "is_vulnerable", "cwe_id", "severity"]
    
    for i, entry in enumerate(data):
        entry_issues = []
        
        # Check required fields
        for field in required_fields:
            if field not in entry:
                entry_issues.append(f"Missing field: {field}")
            elif field == "is_vulnerable":
                # Special handling for boolean field
                if not isinstance(entry[field], bool):
                    entry_issues.append(f"Field {field} must be boolean")
            elif not entry[field]:
                entry_issues.append(f"Empty field: {field}")
        
        # Validate field types
        if "is_vulnerable" in entry and not isinstance(entry["is_vulnerable"], bool):
            entry_issues.append("is_vulnerable must be boolean")
        
        if "cve_ids" in entry and not isinstance(entry["cve_ids"], list):
            entry_issues.append("cve_ids must be array")
        
        # Check code quality
        if "code" in entry:
            code = entry["code"]
            if len(code) < 10:
                entry_issues.append("Code too short (< 10 characters)")
            if len(code) > 5000:
                entry_issues.append("Code too long (> 5000 characters)")
        
        # Check CWE format
        if "cwe_id" in entry:
            cwe = entry["cwe_id"]
            if not cwe.startswith("CWE-") and cwe != "CWE-Unknown":
                entry_issues.append(f"Invalid CWE format: {cwe}")
        
        # Check severity values
        if "severity" in entry:
            severity = entry["severity"]
            if severity not in ["HIGH", "MEDIUM", "LOW"]:
                entry_issues.append(f"Invalid severity: {severity}")
        
        if entry_issues:
            issues.append({
                "entry_index": i,
                "entry_id": entry.get("id", "unknown"),
                "issues": entry_issues
            })
            stats["issues_found"] += len(entry_issues)
        else:
            stats["valid_entries"] += 1
    
    return {
        "statistics": stats,
        "issues": issues[:10],  # Show first 10 issues
        "quality_score": round(stats["valid_entries"] / stats["total_entries"] * 100, 2) if stats["total_entries"] > 0 else 0
    }

def main():
    """Run validation on all datasets"""
    
    base_path = Path("C:/Devign/devign/Custom datasets/data")
    
    datasets = [
        "dataset_combined.json",
        "dataset_devign_mapped.json",
        "dataset_juliet_mapped.json",
        "dataset_github_mapped.json",
        "dataset_owasp_mapped.json"
    ]
    
    for dataset_file in datasets:
        dataset_path = base_path / dataset_file
        if dataset_path.exists():
            print(f"\nValidating {dataset_file}...")
            results = validate_dataset_quality(str(dataset_path))
            
            print(f"Quality Score: {results['quality_score']}%")
            print(f"Valid Entries: {results['statistics']['valid_entries']}/{results['statistics']['total_entries']}")
            print(f"Issues Found: {results['statistics']['issues_found']}")
            
            if results['issues']:
                print("Sample Issues:")
                for issue in results['issues'][:3]:
                    print(f"  Entry {issue['entry_id']}: {', '.join(issue['issues'])}")

if __name__ == "__main__":
    main()
