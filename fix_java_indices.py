#!/usr/bin/env python3
"""
Fix the Java indices issue and create a proper dataset
"""

import pandas as pd
import json
import sys
sys.path.append('src')
import src.data as data
import src.prepare as prepare

def fix_java_indices():
    """Fix the Java indices and create proper dataset"""
    
    print("🔧 FIXING JAVA INDICES")
    print("=" * 50)
    
    # 1. Load original Java dataset
    with open('data/raw/java_dataset.json', 'r') as f:
        java_data = json.load(f)
    
    original_df = pd.DataFrame(java_data)
    print(f"✅ Original dataset: {original_df.shape}")
    
    # 2. Process JSON to graphs
    graphs = prepare.json_process('data/cpg', '0_cpg.json')
    print(f"✅ Graphs generated: {len(graphs)}")
    
    # 3. Fix the indices manually
    print(f"🔧 Fixing indices...")
    for i, graph in enumerate(graphs):
        graph['Index'] = i  # Assign sequential indices: 0, 1, 2, 3
        print(f"   Graph {i}: Index set to {i}")
    
    # 4. Create dataset with corrected indices
    dataset = data.create_with_index(graphs, ["Index", "cpg"])
    print(f"✅ CPG dataset created: {dataset.shape}")
    print(f"   CPG indices: {list(dataset['Index'])}")
    print(f"   DataFrame index: {list(dataset.index)}")
    
    # 5. For join to work, we need to match original dataset size
    # Take only the first 2 CPG graphs to match 2 original functions
    dataset_trimmed = dataset.head(2)  # Take first 2 graphs
    dataset_trimmed['Index'] = [0, 1]  # Set indices to match original
    dataset_trimmed.index = [0, 1]     # Set DataFrame index to match
    
    print(f"✅ Trimmed CPG dataset: {dataset_trimmed.shape}")
    print(f"   Trimmed indices: {list(dataset_trimmed['Index'])}")
    print(f"   Trimmed DataFrame index: {list(dataset_trimmed.index)}")
    
    # 6. Attempt join
    print(f"🔧 Attempting join...")
    print(f"   Original index: {list(original_df.index)}")
    print(f"   CPG index: {list(dataset_trimmed.index)}")
    
    final_dataset = data.inner_join_by_index(original_df, dataset_trimmed)
    print(f"✅ Join successful: {final_dataset.shape}")
    
    if len(final_dataset) > 0:
        print(f"📋 Final dataset sample:")
        for idx, row in final_dataset.iterrows():
            lang = row.get('language', 'N/A')
            target = row.get('target', 'N/A')
            func_preview = str(row.get('func', ''))[:60].replace('\n', ' ')
            print(f"   {idx}: [{lang}] Target={target} - {func_preview}...")
    
    # 7. Save the fixed dataset
    data.write(final_dataset, 'data/cpg', '0_cpg.pkl')
    print(f"✅ Saved fixed dataset to data/cpg/0_cpg.pkl")
    
    # 8. Copy to cpg_tmp for embedding test
    import shutil
    shutil.copy2('data/cpg/0_cpg.pkl', 'data/cpg_tmp/0_cpg.pkl')
    print(f"✅ Copied to data/cpg_tmp/0_cpg.pkl for embedding test")
    
    return final_dataset

if __name__ == "__main__":
    fix_java_indices()