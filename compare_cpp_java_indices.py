#!/usr/bin/env python3
"""
Compare C++ vs Java indices to understand why C++ worked but Java failed
"""

import pandas as pd
import json
import sys
sys.path.append('src')
import src.prepare as prepare

def compare_cpp_java_indices():
    """Compare the index behavior between C++ and Java"""
    
    print("🔍 COMPARING C++ vs JAVA INDICES")
    print("=" * 60)
    
    # 1. Check C++ original dataset
    print("\n📁 C++ Analysis:")
    try:
        with open('data/raw/cpp_dataset.json', 'r') as f:
            cpp_data = json.load(f)
        
        cpp_df = pd.DataFrame(cpp_data)
        print(f"   Original C++ dataset: {cpp_df.shape}")
        print(f"   Original C++ index: {list(cpp_df.index)}")
        
        # Check if there's a C++ JSON file to analyze
        cpp_json_files = ['data/cpg/cpp_cpg.json', 'data/cpg/0_cpg.json']
        cpp_json_file = None
        
        for json_file in cpp_json_files:
            try:
                with open(json_file, 'r') as f:
                    test_data = json.load(f)
                    # Check if this looks like C++ (has 'add' or 'copy' functions)
                    functions = test_data.get('functions', [])
                    func_names = [f.get('function', '') for f in functions]
                    if 'add' in func_names or 'copy' in func_names:
                        cpp_json_file = json_file
                        break
            except:
                continue
        
        if cpp_json_file:
            print(f"   Using C++ JSON: {cpp_json_file}")
            
            # Process C++ graphs
            cpp_graphs = prepare.json_process('data/cpg', cpp_json_file.split('/')[-1])
            if cpp_graphs:
                print(f"   C++ graphs generated: {len(cpp_graphs)}")
                print(f"   C++ graph indices:")
                for i, graph in enumerate(cpp_graphs):
                    index_val = graph.get('Index', 'N/A')
                    func_name = graph.get('func', 'N/A')
                    print(f"     Graph {i}: Index={index_val}, func={func_name}")
            else:
                print(f"   ❌ No C++ graphs generated")
        else:
            print(f"   ❌ No C++ JSON file found")
            
    except Exception as e:
        print(f"   ❌ C++ analysis error: {e}")
    
    # 2. Check Java analysis
    print(f"\n📁 Java Analysis:")
    try:
        with open('data/raw/java_dataset.json', 'r') as f:
            java_data = json.load(f)
        
        java_df = pd.DataFrame(java_data)
        print(f"   Original Java dataset: {java_df.shape}")
        print(f"   Original Java index: {list(java_df.index)}")
        
        # Process Java graphs
        java_graphs = prepare.json_process('data/cpg', '0_cpg.json')
        if java_graphs:
            print(f"   Java graphs generated: {len(java_graphs)}")
            print(f"   Java graph indices:")
            for i, graph in enumerate(java_graphs):
                index_val = graph.get('Index', 'N/A')
                func_name = graph.get('func', 'N/A')
                print(f"     Graph {i}: Index={index_val}, func={func_name}")
        else:
            print(f"   ❌ No Java graphs generated")
            
    except Exception as e:
        print(f"   ❌ Java analysis error: {e}")
    
    # 3. Check if there's a working C++ PKL file
    print(f"\n📁 C++ PKL Analysis:")
    try:
        # Check cpg_tmp first (where we moved the working C++ file)
        cpp_pkl_files = ['data/cpg_tmp/0_cpg.pkl', 'data/cpg/cpp_cpg.pkl']
        
        for pkl_file in cpp_pkl_files:
            try:
                df = pd.read_pickle(pkl_file)
                if len(df) > 0 and 'language' in df.columns:
                    lang = df['language'].iloc[0]
                    if lang == 'cpp':
                        print(f"   Found working C++ PKL: {pkl_file}")
                        print(f"   C++ PKL shape: {df.shape}")
                        print(f"   C++ PKL index: {list(df.index)}")
                        print(f"   C++ PKL 'Index' column: {list(df.get('Index', []))}")
                        break
            except:
                continue
        else:
            print(f"   ❌ No working C++ PKL found")
            
    except Exception as e:
        print(f"   ❌ C++ PKL analysis error: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"🎯 HYPOTHESIS:")
    print(f"   C++ worked because its graph indices were sequential [0,1,2,3]")
    print(f"   Java failed because its graph indices were all the same [8663709,8663709,8663709,8663709]")
    print(f"   This suggests the json_process function behaves differently for different languages")

if __name__ == "__main__":
    compare_cpp_java_indices()