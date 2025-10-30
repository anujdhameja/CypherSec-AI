#!/usr/bin/env python3
"""
Combine all vulnerability datasets and generate comprehensive statistics
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def load_existing_datasets():
    """Load all existing mapped datasets"""
    
    base_path = Path("C:/Devign/devign/Custom datasets/data")
    
    datasets = {}
    dataset_files = [
        "dataset_devign_mapped.json",
        "dataset_juliet_mapped.json", 
        "dataset_github_mapped.json",
        "dataset_owasp_mapped.json"
    ]
    
    for filename in dataset_files:
        filepath = base_path / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    source_name = filename.replace("dataset_", "").replace("_mapped.json", "")
                    datasets[source_name] = data
                    print(f"Loaded {len(data)} entries from {source_name}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"File not found: {filename}")
    
    return datasets

def remove_duplicates(all_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate entries based on code content hash"""
    
    seen_hashes = set()
    unique_data = []
    duplicates_removed = 0
    
    for entry in all_data:
        # Create hash from code content and function name
        content = f"{entry.get('code', '')}{entry.get('function', '')}"
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_data.append(entry)
        else:
            duplicates_removed += 1
    
    print(f"Removed {duplicates_removed} duplicate entries")
    return unique_data

def generate_comprehensive_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive dataset statistics"""
    
    # Basic counts
    total_entries = len(data)
    vulnerable_entries = sum(1 for entry in data if entry.get("is_vulnerable", False))
    safe_entries = total_entries - vulnerable_entries
    
    # Source distribution
    source_counts = {}
    for entry in data:
        source = entry.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # CWE distribution
    cwe_counts = {}
    for entry in data:
        cwe_id = entry.get("cwe_id", "Unknown")
        cwe_counts[cwe_id] = cwe_counts.get(cwe_id, 0) + 1
    
    # Severity distribution
    severity_counts = {}
    for entry in data:
        severity = entry.get("severity", "Unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    # Vulnerability by CWE
    vuln_by_cwe = {}
    for entry in data:
        if entry.get("is_vulnerable", False):
            cwe_id = entry.get("cwe_id", "Unknown")
            vuln_by_cwe[cwe_id] = vuln_by_cwe.get(cwe_id, 0) + 1
    
    # Function name analysis
    function_names = {}
    for entry in data:
        func_name = entry.get("function", "unknown")
        function_names[func_name] = function_names.get(func_name, 0) + 1
    
    # Code length statistics
    code_lengths = [len(entry.get("code", "")) for entry in data]
    avg_code_length = sum(code_lengths) / len(code_lengths) if code_lengths else 0
    
    # CVE information
    entries_with_cve = sum(1 for entry in data if entry.get("cve_ids") and len(entry["cve_ids"]) > 0)
    
    return {
        "summary": {
            "total_entries": total_entries,
            "vulnerable_entries": vulnerable_entries,
            "safe_entries": safe_entries,
            "vulnerability_rate": round(vulnerable_entries / total_entries * 100, 2) if total_entries > 0 else 0,
            "entries_with_cve": entries_with_cve
        },
        "source_distribution": dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)),
        "cwe_distribution": dict(sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)),
        "severity_distribution": dict(sorted(severity_counts.items(), key=lambda x: x[1], reverse=True)),
        "vulnerability_by_cwe": dict(sorted(vuln_by_cwe.items(), key=lambda x: x[1], reverse=True)),
        "code_statistics": {
            "average_code_length": round(avg_code_length, 2),
            "min_code_length": min(code_lengths) if code_lengths else 0,
            "max_code_length": max(code_lengths) if code_lengths else 0
        },
        "top_function_names": dict(sorted(function_names.items(), key=lambda x: x[1], reverse=True)[:10]),
        "generated_at": datetime.now().isoformat()
    }

def create_dataset_summary_report(stats: Dict[str, Any]) -> str:
    """Create a human-readable summary report"""
    
    report = []
    report.append("=" * 60)
    report.append("VULNERABILITY DATASET COLLECTION SUMMARY")
    report.append("=" * 60)
    report.append("")
    
    # Summary
    summary = stats["summary"]
    report.append(f"📊 DATASET OVERVIEW")
    report.append(f"   Total Entries: {summary['total_entries']:,}")
    report.append(f"   Vulnerable: {summary['vulnerable_entries']:,} ({summary['vulnerability_rate']}%)")
    report.append(f"   Safe: {summary['safe_entries']:,}")
    report.append(f"   With CVE IDs: {summary['entries_with_cve']:,}")
    report.append("")
    
    # Source breakdown
    report.append(f"📁 SOURCE DISTRIBUTION")
    for source, count in stats["source_distribution"].items():
        percentage = round(count / summary['total_entries'] * 100, 1)
        report.append(f"   {source.capitalize()}: {count:,} ({percentage}%)")
    report.append("")
    
    # CWE breakdown
    report.append(f"🔍 TOP VULNERABILITY TYPES (CWE)")
    for i, (cwe, count) in enumerate(list(stats["cwe_distribution"].items())[:8]):
        percentage = round(count / summary['total_entries'] * 100, 1)
        report.append(f"   {i+1}. {cwe}: {count:,} ({percentage}%)")
    report.append("")
    
    # Severity breakdown
    report.append(f"⚠️  SEVERITY DISTRIBUTION")
    for severity, count in stats["severity_distribution"].items():
        percentage = round(count / summary['total_entries'] * 100, 1)
        report.append(f"   {severity}: {count:,} ({percentage}%)")
    report.append("")
    
    # Code statistics
    code_stats = stats["code_statistics"]
    report.append(f"📝 CODE STATISTICS")
    report.append(f"   Average Length: {code_stats['average_code_length']:,.0f} characters")
    report.append(f"   Range: {code_stats['min_code_length']:,} - {code_stats['max_code_length']:,} characters")
    report.append("")
    
    # Most vulnerable CWEs
    report.append(f"🚨 MOST CRITICAL VULNERABILITIES")
    for i, (cwe, count) in enumerate(list(stats["vulnerability_by_cwe"].items())[:5]):
        report.append(f"   {i+1}. {cwe}: {count:,} vulnerable instances")
    report.append("")
    
    report.append(f"Generated: {stats['generated_at']}")
    report.append("=" * 60)
    
    return "\n".join(report)

def main():
    """Main execution function"""
    
    print("Combining all vulnerability datasets...")
    
    # Load all datasets
    datasets = load_existing_datasets()
    
    if not datasets:
        print("No datasets found to combine!")
        return
    
    # Combine all data
    all_data = []
    for source, data in datasets.items():
        all_data.extend(data)
        print(f"Added {len(data)} entries from {source}")
    
    print(f"Total entries before deduplication: {len(all_data)}")
    
    # Remove duplicates
    unique_data = remove_duplicates(all_data)
    print(f"Total unique entries: {len(unique_data)}")
    
    # Save combined dataset
    base_path = Path("C:/Devign/devign/Custom datasets/data")
    combined_path = base_path / "dataset_combined.json"
    
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved combined dataset to: {combined_path}")
    
    # Generate statistics
    stats = generate_comprehensive_statistics(unique_data)
    
    # Save statistics
    stats_path = base_path / "dataset_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Saved statistics to: {stats_path}")
    
    # Create and save summary report
    report = create_dataset_summary_report(stats)
    report_path = base_path / "dataset_summary_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Saved summary report to: {report_path}")
    
    # Print summary to console
    print("\n" + report)
    
    return stats

if __name__ == "__main__":
    main()