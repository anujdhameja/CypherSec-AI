#!/usr/bin/env python3
"""
Debug the Java pipeline to see where it's failing
"""

import pandas as pd
import json
import os

def debug_java_pipeline():
    """Debug each step of the Java pipeline"""
    
    print("🔍 DEBUGGING JAVA PIPELINE")
    print("=" * 50)
    
    # 1. Check original Java dataset
    print("\n📁 1. Original Java Dataset:")
    try:
        with open('data/raw/java_dataset.json', 'r') as f:
            java_data = json.load(f)
        
        print(f"   Samples: {len(java_data)}")
        for i, sample in enumerate(java_data):
            target_label = "VULNERABLE" if sample['target'] == 1 else "SAFE"
            func_preview = sample['func'][:80].replace('\n', ' ')
            print(f"   {i}: [{target_label}] {func_preview}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 2. Check Java JSON from Joern
    print("\n📁 2. Joern JSON Output:")
    try:
        with open('data/cpg/0_cpg.json', 'r') as f:
            json_data = json.load(f)
        
        functions = json_data.get('functions', [])
        print(f"   Functions found: {len(functions)}")
        for i, func in enumerate(functions):
            func_name = func.get('function', 'N/A')
            file_name = func.get('file', 'N/A')
            node_count = len(func.get('nodes', []))
            print(f"   {i}: '{func_name}' in {file_name} ({node_count} nodes)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Check main CPG PKL file
    print("\n📁 3. Main CPG PKL File:")
    try:
        df = pd.read_pickle('data/cpg/0_cpg.pkl')
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"   Sample data:")
            for idx, row in df.iterrows():
                lang = row.get('language', 'N/A')
                target = row.get('target', 'N/A')
                func_preview = str(row.get('func', ''))[:60].replace('\n', ' ')
                print(f"     {idx}: [{lang}] Target={target} - {func_preview}...")
        else:
            print(f"   ❌ Main PKL file is empty!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 4. Check cpg_tmp PKL file
    print("\n📁 4. CPG_TMP PKL File:")
    try:
        df = pd.read_pickle('data/cpg_tmp/0_cpg.pkl')
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"   Sample data:")
            for idx, row in df.iterrows():
                lang = row.get('language', 'N/A')
                target = row.get('target', 'N/A')
                func_preview = str(row.get('func', ''))[:60].replace('\n', ' ')
                print(f"     {idx}: [{lang}] Target={target} - {func_preview}...")
        else:
            print(f"   ❌ CPG_TMP PKL file is empty!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Check if files are being overwritten
    print("\n📁 5. File Timestamps:")
    files_to_check = [
        'data/raw/java_dataset.json',
        'data/cpg/0_cpg.json', 
        'data/cpg/0_cpg.pkl',
        'data/cpg_tmp/0_cpg.pkl'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            import datetime
            timestamp = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   {file_path}: {timestamp}")
        else:
            print(f"   {file_path}: NOT FOUND")

if __name__ == "__main__":
    debug_java_pipeline()