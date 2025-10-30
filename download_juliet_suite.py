#!/usr/bin/env python3
"""
Download and process the NIST Juliet Test Suite
"""

import os
import requests
import zipfile
import json
import hashlib
from pathlib import Path
import re
from typing import List, Dict, Any

def download_juliet_test_suite():
    """Download the Juliet Test Suite from NIST"""
    
    base_path = Path("C:/Devign/devign/Custom datasets/data")
    juliet_dir = base_path / "juliet_raw"
    juliet_dir.mkdir(parents=True, exist_ok=True)
    
    # Juliet Test Suite URL (C/C++ version)
    url = "https://samate.nist.gov/SARD/downloads/test-suites/juliet-test-suite-v1.3-for-c-cpp.zip"
    zip_path = juliet_dir / "juliet-test-suite-v1.3.zip"
    
    print("Downloading Juliet Test Suite...")
    print(f"URL: {url}")
    print(f"Destination: {zip_path}")
    
    if zip_path.exists():
        print("Juliet Test Suite already downloaded.")
        return str(zip_path)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        print(f"\nDownload completed: {zip_path}")
        return str(zip_path)
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download Juliet Test Suite: {e}")
        print("You can manually download from: https://samate.nist.gov/SARD/")
        return None

def extract_juliet_suite(zip_path: str):
    """Extract the Juliet Test Suite"""
    
    if not zip_path or not Path(zip_path).exists():
        print("Zip file not found, skipping extraction")
        return None
    
    extract_dir = Path(zip_path).parent / "extracted"
    
    if extract_dir.exists():
        print("Juliet Test Suite already extracted.")
        return str(extract_dir)
    
    print(f"Extracting to: {extract_dir}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed.")
        return str(extract_dir)
        
    except Exception as e:
        print(f"Failed to extract: {e}")
        return None

def process_juliet_files(extract_dir: str, max_files: int = 200):
    """Process extracted Juliet files and create mappings"""
    
    if not extract_dir or not Path(extract_dir).exists():
        print("Extract directory not found")
        return []
    
    extract_path = Path(extract_dir)
    
    # Find C/C++ files
    c_files = list(extract_path.rglob("*.c")) + list(extract_path.rglob("*.cpp"))
    print(f"Found {len(c_files)} C/C++ files")
    
    if len(c_files) > max_files:
        print(f"Limiting to first {max_files} files for processing")
        c_files = c_files[:max_files]
    
    mapped_data = []
    
    for i, file_path in enumerate(c_files):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
            
            # Skip very large files
            if len(code_content) > 10000:
                continue
            
            # Extract info from path and filename
            relative_path = file_path.relative_to(extract_path)
            path_str = str(relative_path)
            
            # Determine if vulnerable based on filename/path
            is_vulnerable = (
                "bad" in file_path.name.lower() or
                "vuln" in path_str.lower() or
                "flaw" in path_str.lower()
            )
            
            # Extract CWE from path
            cwe_id = extract_cwe_from_path(path_str)
            
            # Extract function name
            function_name = extract_main_function(code_content)
            
            # Determine severity
            severity = "HIGH" if is_vulnerable else "LOW"
            if cwe_id in ["CWE-190", "CWE-401"]:  # Integer overflow, memory leak
                severity = "MEDIUM" if is_vulnerable else "LOW"
            
            entry = {
                "id": f"juliet_{hashlib.md5(path_str.encode()).hexdigest()[:8]}",
                "source": "juliet",
                "file": path_str,
                "function": function_name,
                "code": code_content[:2000],  # Limit code length
                "is_vulnerable": is_vulnerable,
                "cwe_id": cwe_id,
                "cve_ids": [],
                "severity": severity
            }
            
            mapped_data.append(entry)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(c_files)} files")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Successfully processed {len(mapped_data)} files")
    return mapped_data

def extract_cwe_from_path(path_str: str) -> str:
    """Extract CWE ID from file path"""
    
    # Look for CWE pattern in path
    cwe_match = re.search(r'CWE[_-]?(\d+)', path_str, re.IGNORECASE)
    if cwe_match:
        return f"CWE-{cwe_match.group(1)}"
    
    # Map based on keywords in path
    path_lower = path_str.lower()
    keyword_mappings = {
        "buffer": "CWE-119",
        "overflow": "CWE-119", 
        "sql": "CWE-89",
        "injection": "CWE-89",
        "xss": "CWE-79",
        "cross_site": "CWE-79",
        "path": "CWE-22",
        "traversal": "CWE-22",
        "command": "CWE-78",
        "integer": "CWE-190",
        "use_after_free": "CWE-416",
        "null_pointer": "CWE-476",
        "memory_leak": "CWE-401",
        "race": "CWE-362"
    }
    
    for keyword, cwe_id in keyword_mappings.items():
        if keyword in path_lower:
            return cwe_id
    
    return "CWE-Unknown"

def extract_main_function(code: str) -> str:
    """Extract the main function name from code"""
    
    # Look for function definitions
    func_patterns = [
        r'void\s+(\w+)\s*\(',
        r'int\s+(\w+)\s*\(',
        r'char\s*\*\s*(\w+)\s*\(',
        r'static\s+void\s+(\w+)\s*\('
    ]
    
    for pattern in func_patterns:
        matches = re.findall(pattern, code)
        if matches:
            # Return first non-main function
            for func in matches:
                if func not in ["main", "printf", "scanf", "malloc", "free"]:
                    return func
            return matches[0]
    
    return "unknown_function"

def main():
    """Main execution"""
    
    print("Starting Juliet Test Suite download and processing...")
    
    # Download
    zip_path = download_juliet_test_suite()
    if not zip_path:
        print("Download failed, creating placeholder data")
        return
    
    # Extract
    extract_dir = extract_juliet_suite(zip_path)
    if not extract_dir:
        print("Extraction failed")
        return
    
    # Process files
    mapped_data = process_juliet_files(extract_dir)
    
    if not mapped_data:
        print("No data processed")
        return
    
    # Save results
    output_path = Path("C:/Devign/devign/Custom datasets/data/dataset_juliet_mapped.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapped_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(mapped_data)} entries to {output_path}")
    
    # Print statistics
    vulnerable_count = sum(1 for entry in mapped_data if entry["is_vulnerable"])
    safe_count = len(mapped_data) - vulnerable_count
    
    cwe_counts = {}
    for entry in mapped_data:
        cwe_id = entry.get("cwe_id", "Unknown")
        cwe_counts[cwe_id] = cwe_counts.get(cwe_id, 0) + 1
    
    print(f"\nJuliet Dataset Statistics:")
    print(f"Total entries: {len(mapped_data)}")
    print(f"Vulnerable: {vulnerable_count}")
    print(f"Safe: {safe_count}")
    print(f"\nTop CWE types:")
    for cwe, count in sorted(cwe_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cwe}: {count}")

if __name__ == "__main__":
    main()