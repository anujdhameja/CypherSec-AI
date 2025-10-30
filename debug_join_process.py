#!/usr/bin/env python3
"""
Debug the join process specifically for Java
"""

import pandas as pd
import json
import sys
sys.path.append('src')
import src.data as data
import src.prepare as prepare

def debug_join_process():
    """Debug the join process step by step"""
    
    print("🔍 DEBUGGING JOIN PROCESS")
    print("=" * 50)
    
    # 1. Load original Java dataset
    print("\n📁 1. Loading Original Java Dataset:")
    with open('data/raw/java_dataset.json', 'r') as f:
        java_data = json.load(f)
    
    original_df = pd.DataFrame(java_data)
    print(f"   Shape: {original_df.shape}")
    print(f"   Columns: {list(original_df.columns)}")
    print(f"   Index: {list(original_df.index)}")
    
    # 2. Process JSON to graphs
    print("\n📁 2. Processing JSON to Graphs:")
    graphs = prepare.json_process('data/cpg', '0_cpg.json')
    
    if graphs is None:
        print("   ❌ No graphs generated!")
        return
    
    print(f"   Graphs generated: {len(graphs)}")
    for i, graph in enumerate(graphs):
        print(f"     Graph {i}: {list(graph.keys())}")
    
    # 3. Create dataset with index
    print("\n📁 3. Creating Dataset with Index:")
    try:
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        print(f"   CPG Dataset shape: {dataset.shape}")
        print(f"   CPG Dataset columns: {list(dataset.columns)}")
        print(f"   CPG Dataset index: {list(dataset.index)}")
        print(f"   CPG Dataset 'Index' column: {list(dataset['Index'])}")
    except Exception as e:
        print(f"   ❌ Error creating dataset: {e}")
        return
    
    # 4. Attempt join
    print("\n📁 4. Attempting Join:")
    try:
        print(f"   Original DF index: {list(original_df.index)}")
        print(f"   CPG Dataset 'Index': {list(dataset['Index'])}")
        
        # Try the join
        final_dataset = data.inner_join_by_index(original_df, dataset)
        print(f"   ✅ Join successful!")
        print(f"   Final dataset shape: {final_dataset.shape}")
        print(f"   Final dataset columns: {list(final_dataset.columns)}")
        
        if len(final_dataset) > 0:
            print(f"   Sample data:")
            for idx, row in final_dataset.iterrows():
                lang = row.get('language', 'N/A')
                target = row.get('target', 'N/A')
                print(f"     {idx}: [{lang}] Target={target}")
        
        # Save the corrected dataset
        print(f"\n📁 5. Saving Corrected Dataset:")
        data.write(final_dataset, 'data/cpg', '0_cpg.pkl')
        print(f"   ✅ Saved corrected dataset to data/cpg/0_cpg.pkl")
        
    except Exception as e:
        print(f"   ❌ Join failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_join_process()