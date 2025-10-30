# Vulnerability Dataset Collection

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
