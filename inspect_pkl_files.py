#!/usr/bin/env python3
"""
Inspect the generated PKL files to see their contents
"""

import pandas as pd
import os

def inspect_pkl_files():
    """Inspect all PKL files in cpg_tmp directory"""
    
    cpg_tmp_dir = "data/cpg_tmp"
    pkl_files = ['cpp_cpg.pkl', 'csharp_cpg.pkl', 'java_cpg.pkl', 'python_cpg.pkl']
    
    print("🔍 INSPECTING PKL FILES")
    print("=" * 60)
    
    for pkl_file in pkl_files:
        pkl_path = os.path.join(cpg_tmp_dir, pkl_file)
        
        if os.path.exists(pkl_path):
            print(f"\n📁 {pkl_file}")
            print("-" * 40)
            
            try:
                # Load the PKL file
                df = pd.read_pickle(pkl_path)
                
                print(f"📊 Shape: {df.shape}")
                print(f"📋 Columns: {list(df.columns)}")
                
                if len(df) > 0:
                    print(f"🎯 Sample data:")
                    sample = df.iloc[0]
                    
                    # Show basic info
                    print(f"  Language: {sample.get('language', 'N/A')}")
                    print(f"  Target: {sample.get('target', 'N/A')}")
                    print(f"  Project: {sample.get('project', 'N/A')}")
                    print(f"  Commit ID: {sample.get('commit_id', 'N/A')}")
                    
                    # Show function code (first 100 chars)
                    func_code = str(sample.get('func', 'N/A'))
                    if len(func_code) > 100:
                        func_code = func_code[:100] + "..."
                    print(f"  Function: {func_code}")
                    
                    # Show CPG info
                    cpg_data = sample.get('cpg', {})
                    if isinstance(cpg_data, dict):
                        print(f"  CPG keys: {list(cpg_data.keys())}")
                        if 'nodes' in cpg_data:
                            nodes = cpg_data['nodes']
                            print(f"  CPG nodes: {len(nodes) if hasattr(nodes, '__len__') else 'N/A'}")
                        if 'edges' in cpg_data:
                            edges = cpg_data['edges']
                            print(f"  CPG edges: {len(edges) if hasattr(edges, '__len__') else 'N/A'}")
                    else:
                        print(f"  CPG type: {type(cpg_data)}")
                    
                    # Show all samples in this file
                    print(f"\n  📋 All samples in {pkl_file}:")
                    for idx, row in df.iterrows():
                        target_label = "VULNERABLE" if row.get('target') == 1 else "SAFE"
                        func_preview = str(row.get('func', ''))[:50].replace('\n', ' ')
                        print(f"    {idx}: [{target_label}] {func_preview}...")
                        
                else:
                    print("  ❌ No data in this file")
                    
            except Exception as e:
                print(f"  ❌ Error reading {pkl_file}: {e}")
        else:
            print(f"\n❌ {pkl_file} not found")
    
    print(f"\n" + "=" * 60)
    print("🎯 SUMMARY:")
    
    total_samples = 0
    languages_with_data = []
    
    for pkl_file in pkl_files:
        pkl_path = os.path.join(cpg_tmp_dir, pkl_file)
        if os.path.exists(pkl_path):
            try:
                df = pd.read_pickle(pkl_path)
                lang = pkl_file.replace('_cpg.pkl', '')
                sample_count = len(df)
                total_samples += sample_count
                
                if sample_count > 0:
                    languages_with_data.append(lang)
                    vuln_count = df['target'].sum() if 'target' in df.columns else 0
                    safe_count = sample_count - vuln_count
                    print(f"  {lang.upper()}: {sample_count} samples ({vuln_count} vuln, {safe_count} safe)")
                else:
                    print(f"  {lang.upper()}: 0 samples (EMPTY)")
                    
            except Exception as e:
                print(f"  {pkl_file}: Error - {e}")
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Languages with data: {languages_with_data}")
    
    if total_samples > 0:
        print(f"\n✅ Ready for embedding step: python main.py -e")
    else:
        print(f"\n❌ No samples found - need to fix the PKL generation")

if __name__ == "__main__":
    inspect_pkl_files()