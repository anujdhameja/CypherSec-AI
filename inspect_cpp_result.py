#!/usr/bin/env python3
"""
Inspect the C++ result to understand the 4 graphs issue
"""

import pandas as pd
import json

def inspect_cpp_result():
    """Inspect the C++ processing result"""
    
    print("🔍 INSPECTING C++ PROCESSING RESULT")
    print("=" * 50)
    
    # 1. Check the original C++ dataset
    print("\n📁 1. Original C++ Dataset:")
    with open('data/raw/cpp_dataset.json', 'r') as f:
        cpp_data = json.load(f)
    
    print(f"   Samples: {len(cpp_data)}")
    for i, sample in enumerate(cpp_data):
        target_label = "VULNERABLE" if sample['target'] == 1 else "SAFE"
        func_preview = sample['func'][:60].replace('\n', ' ')
        print(f"   {i}: [{target_label}] {func_preview}...")
    
    # 2. Check the JSON file to see what functions Joern found
    print("\n📁 2. Joern JSON Output:")
    with open('data/cpg/0_cpg.json', 'r') as f:
        json_data = json.load(f)
    
    functions = json_data.get('functions', [])
    print(f"   Functions found: {len(functions)}")
    for i, func in enumerate(functions):
        func_name = func.get('function', 'N/A')
        file_name = func.get('file', 'N/A')
        node_count = len(func.get('nodes', []))
        print(f"   {i}: '{func_name}' in {file_name} ({node_count} nodes)")
    
    # 3. Check the final PKL file
    print("\n📁 3. Final PKL Dataset:")
    df = pd.read_pickle('data/cpg/0_cpg.pkl')
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"\n   📋 All samples in PKL:")
        for idx, row in df.iterrows():
            target_label = "VULNERABLE" if row.get('target') == 1 else "SAFE"
            lang = row.get('language', 'N/A')
            func_preview = str(row.get('func', ''))[:50].replace('\n', ' ')
            
            # Check CPG data
            cpg_data = row.get('cpg', {})
            if isinstance(cpg_data, dict):
                nodes_count = len(cpg_data.get('nodes', []))
                edges_count = len(cpg_data.get('edges', []))
                cpg_info = f"({nodes_count} nodes, {edges_count} edges)"
            else:
                cpg_info = f"(type: {type(cpg_data)})"
            
            print(f"   {idx}: [{target_label}] {lang} - {func_preview}... {cpg_info}")
    
    # 4. Analysis
    print(f"\n📊 ANALYSIS:")
    print(f"   Original functions: 2")
    print(f"   Joern found functions: {len(functions)}")
    print(f"   Final dataset samples: {len(df)}")
    
    if len(functions) > 2:
        print(f"\n   🔍 Extra functions found by Joern:")
        for i, func in enumerate(functions):
            func_name = func.get('function', 'N/A')
            if func_name not in ['add', 'copy']:
                print(f"   - '{func_name}' (likely implicit/generated)")
    
    print(f"\n   💡 This is normal - Joern finds ALL functions in the code,")
    print(f"      including implicit ones like global initialization, etc.")

if __name__ == "__main__":
    inspect_cpp_result()