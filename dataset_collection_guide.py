#!/usr/bin/env python3
"""
Comprehensive Vulnerability Dataset Collection Guide and Framework
"""

import json
from pathlib import Path
from datetime import datetime

def create_collection_guide():
    """Create comprehensive guide for dataset collection"""
    
    guide = {
        "dataset_collection_guide": {
            "overview": "Comprehensive vulnerability dataset collection for machine learning training",
            "target_location": "C:/Devign/devign/Custom datasets/data/",
            "estimated_time": "3-4 hours (including downloads)",
            "datasets": {
                "devign": {
                    "name": "Devign Dataset",
                    "status": "Already available",
                    "description": "Graph-based vulnerability detection dataset",
                    "source": "Local repository",
                    "entries_expected": "27,000+",
                    "action": "Map existing data to standard format"
                },
                "juliet": {
                    "name": "NIST Juliet Test Suite",
                    "status": "Manual download required",
                    "description": "Comprehensive test cases for static analysis tools",
                    "source": "https://samate.nist.gov/SARD/",
                    "direct_url": "https://samate.nist.gov/SARD/downloads/test-suites/juliet-test-suite-v1.3-for-c-cpp.zip",
                    "entries_expected": "64,000+",
                    "file_size": "~500MB",
                    "manual_steps": [
                        "1. Visit https://samate.nist.gov/SARD/",
                        "2. Navigate to Test Suites section",
                        "3. Download 'Juliet Test Suite v1.3 for C/C++'",
                        "4. Save to: C:/Devign/devign/Custom datasets/data/juliet_raw/",
                        "5. Run processing script"
                    ],
                    "action": "Download, extract, and process C/C++ test cases"
                },
                "github_vuln": {
                    "name": "GitHub Vulnerability Examples",
                    "status": "Automated collection",
                    "description": "Real-world vulnerability patterns from GitHub",
                    "source": "GitHub repositories and security advisories",
                    "entries_expected": "100+",
                    "action": "Collect using GitHub API and manual curation"
                },
                "owasp": {
                    "name": "OWASP Vulnerability Examples",
                    "status": "Automated collection",
                    "description": "OWASP Top 10 and WebGoat examples",
                    "source": "OWASP projects and documentation",
                    "entries_expected": "50+",
                    "action": "Extract from OWASP resources"
                },
                "additional_sources": {
                    "cve_database": {
                        "name": "CVE Database Examples",
                        "source": "https://cve.mitre.org/",
                        "description": "Real CVE examples with code snippets"
                    },
                    "exploit_db": {
                        "name": "Exploit Database",
                        "source": "https://www.exploit-db.com/",
                        "description": "Proof-of-concept exploits"
                    },
                    "security_advisories": {
                        "name": "Security Advisories",
                        "source": "Various vendors",
                        "description": "Vendor security advisories with code examples"
                    }
                }
            },
            "mapping_format": {
                "description": "Standard format for all datasets",
                "required_fields": {
                    "id": "Unique identifier (string)",
                    "source": "Dataset source name (string)",
                    "file": "Original file path (string)",
                    "function": "Function name containing the code (string)",
                    "code": "Source code snippet (string, max 2000 chars)",
                    "is_vulnerable": "Vulnerability flag (boolean)",
                    "cwe_id": "CWE identifier (string, e.g., 'CWE-119')",
                    "cve_ids": "List of related CVE IDs (array of strings)",
                    "severity": "Severity level (string: HIGH/MEDIUM/LOW)"
                },
                "optional_fields": {
                    "description": "Human-readable description",
                    "language": "Programming language",
                    "category": "Vulnerability category",
                    "fix_suggestion": "How to fix the vulnerability"
                }
            },
            "processing_steps": [
                "1. Download/collect raw datasets",
                "2. Extract and parse source files",
                "3. Map to standard format",
                "4. Remove duplicates",
                "5. Validate data quality",
                "6. Generate statistics",
                "7. Create combined dataset"
            ]
        }
    }
    
    return guide

def create_manual_download_instructions():
    """Create detailed manual download instructions"""
    
    instructions = """
# MANUAL DATASET DOWNLOAD INSTRUCTIONS

## 1. NIST Juliet Test Suite (REQUIRED)

### Download Steps:
1. Open browser and go to: https://samate.nist.gov/SARD/
2. Click on "Test Suites" in the navigation
3. Find "Juliet Test Suite v1.3 for C/C++"
4. Click download link (juliet-test-suite-v1.3-for-c-cpp.zip)
5. Save to: C:/Devign/devign/Custom datasets/data/juliet_raw/

### Alternative URLs (if main site is down):
- Mirror 1: https://github.com/NIST-SARD/juliet-test-suite-cplusplus
- Mirror 2: Search for "NIST Juliet Test Suite" on academic repositories

### File Details:
- Size: ~500MB compressed, ~2GB extracted
- Contains: 64,000+ test cases for C/C++
- Format: Individual .c and .cpp files organized by CWE type

## 2. GitHub Vulnerability Database (AUTOMATED)

### Collection Method:
- GitHub API search for repositories with vulnerability keywords
- Security advisory database
- Manual curation of high-quality examples

### Search Queries Used:
- "cwe-119 vulnerable language:c"
- "buffer overflow vulnerable language:c"
- "sql injection vulnerable language:c"
- "use after free vulnerable language:c"

## 3. OWASP Examples (AUTOMATED)

### Sources:
- OWASP WebGoat project
- OWASP Top 10 documentation
- OWASP Code Review Guide examples

## 4. Additional Sources (OPTIONAL)

### CVE Database:
- Visit: https://cve.mitre.org/
- Search for CVEs with code examples
- Focus on C/C++ vulnerabilities

### Exploit Database:
- Visit: https://www.exploit-db.com/
- Filter by platform: C/C++
- Look for proof-of-concept code

## PROCESSING AFTER DOWNLOAD

1. Run: python download_juliet_suite.py
2. Run: python collect_github_vulnerabilities.py  
3. Run: python combine_all_datasets.py

## EXPECTED RESULTS

After successful collection:
- dataset_devign_mapped.json: ~2,000 entries
- dataset_juliet_mapped.json: ~10,000 entries (sample)
- dataset_github_mapped.json: ~100 entries
- dataset_owasp_mapped.json: ~50 entries
- dataset_combined.json: ~12,000 unique entries

Total estimated dataset size: 10,000-15,000 vulnerability examples
"""
    
    return instructions

def create_quality_validation_script():
    """Create script to validate dataset quality"""
    
    validation_script = '''#!/usr/bin/env python3
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
            if field not in entry or not entry[field]:
                entry_issues.append(f"Missing or empty field: {field}")
        
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
            print(f"\\nValidating {dataset_file}...")
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
'''
    
    return validation_script

def main():
    """Create all guide files"""
    
    base_path = Path("C:/Devign/devign/Custom datasets")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create collection guide
    guide = create_collection_guide()
    guide_path = base_path / "dataset_collection_guide.json"
    with open(guide_path, 'w', encoding='utf-8') as f:
        json.dump(guide, f, indent=2, ensure_ascii=False)
    
    print(f"Created collection guide: {guide_path}")
    
    # Create manual download instructions
    instructions = create_manual_download_instructions()
    instructions_path = base_path / "MANUAL_DOWNLOAD_INSTRUCTIONS.md"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"Created download instructions: {instructions_path}")
    
    # Create validation script
    validation_script = create_quality_validation_script()
    validation_path = base_path / "validate_dataset_quality.py"
    with open(validation_path, 'w', encoding='utf-8') as f:
        f.write(validation_script)
    
    print(f"Created validation script: {validation_path}")
    
    # Create README
    readme_content = """# Vulnerability Dataset Collection

This directory contains scripts and data for collecting and mapping vulnerability datasets for machine learning training.

## Quick Start

1. Run `python create_example_datasets.py` - Creates example datasets
2. Run `python collect_github_vulnerabilities.py` - Collects GitHub examples  
3. Run `python combine_all_datasets.py` - Combines all datasets
4. Check `data/dataset_summary_report.txt` for results

## Manual Downloads Required

- **Juliet Test Suite**: See `MANUAL_DOWNLOAD_INSTRUCTIONS.md`
- Download from: https://samate.nist.gov/SARD/

## Files Created

- `data/dataset_combined.json` - All datasets combined
- `data/dataset_statistics.json` - Comprehensive statistics
- `data/dataset_summary_report.txt` - Human-readable summary

## Dataset Format

Each entry follows this structure:
```json
{
  "id": "unique_id",
  "source": "dataset_name", 
  "file": "path/to/file.c",
  "function": "function_name",
  "code": "source code...",
  "is_vulnerable": true,
  "cwe_id": "CWE-119",
  "cve_ids": ["CVE-2015-0001"],
  "severity": "HIGH"
}
```

## Quality Validation

Run `python validate_dataset_quality.py` to check data quality.

## Current Status

✅ Example datasets created
✅ GitHub vulnerability examples collected  
✅ OWASP examples collected
⏳ Juliet Test Suite (manual download required)
✅ Combined dataset with statistics
"""
    
    readme_path = base_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Created README: {readme_path}")
    
    print("\n" + "="*60)
    print("DATASET COLLECTION FRAMEWORK CREATED")
    print("="*60)
    print(f"Location: {base_path}")
    print("\nFiles created:")
    print("- dataset_collection_guide.json")
    print("- MANUAL_DOWNLOAD_INSTRUCTIONS.md") 
    print("- validate_dataset_quality.py")
    print("- README.md")
    print("\nNext steps:")
    print("1. Review MANUAL_DOWNLOAD_INSTRUCTIONS.md")
    print("2. Download Juliet Test Suite manually")
    print("3. Run processing scripts")
    print("4. Validate dataset quality")

if __name__ == "__main__":
    main()