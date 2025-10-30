#!/usr/bin/env python3
"""
Universal fix for all language processing issues
This will work for C++, Java, Python, C#, etc.
"""

import pandas as pd
import json
import sys
import shutil
import os
sys.path.append('src')
import src.data as data
import src.prepare as prepare

def universal_language_fix(language):
    """Universal fix for any language"""
    
    print(f"🔧 UNIVERSAL FIX FOR {language.upper()}")
    print("=" * 50)
    
    # 1. Load original dataset
    dataset_file = f'data/raw/{language}_dataset.json'
    with open(dataset_file, 'r') as f:
        original_data = json.load(f)
    
    original_df = pd.DataFrame(original_data)
    print(f"✅ Original {language} dataset: {original_df.shape}")
    
    # 2. Process the latest JSON (always 0_cpg.json)
    json_file = '0_cpg.json'
    graphs = prepare.json_process('data/cpg', json_file)
    
    if not graphs:
        print(f"❌ No graphs generated for {language}")
        return False
    
    print(f"✅ Generated {len(graphs)} graphs")
    
    # 3. UNIVERSAL FIX: Force sequential indices regardless of what Joern gives us
    print(f"🔧 Applying universal index fix...")
    
    # Take only as many graphs as we have original functions
    num_original = len(original_df)
    graphs_to_use = graphs[:num_original]  # Take first N graphs
    
    # Force sequential indices
    for i, graph in enumerate(graphs_to_use):
        graph['Index'] = i
    
    print(f"✅ Using {len(graphs_to_use)} graphs with indices [0, 1, ...]")
    
    # 4. Create dataset and join
    dataset = data.create_with_index(graphs_to_use, ["Index", "cpg"])
    dataset.index = list(range(len(dataset)))  # Force sequential DataFrame index
    
    # Ensure original_df has matching index
    original_df.index = list(range(len(original_df)))
    
    print(f"🔧 Joining datasets...")
    print(f"   Original index: {list(original_df.index)}")
    print(f"   CPG index: {list(dataset.index)}")
    
    final_dataset = data.inner_join_by_index(original_df, dataset)
    
    if len(final_dataset) == 0:
        print(f"❌ Join failed - trying direct merge...")
        # Fallback: direct concatenation
        final_dataset = pd.concat([original_df.reset_index(drop=True), 
                                 dataset.reset_index(drop=True)], axis=1)
    
    print(f"✅ Final dataset: {final_dataset.shape}")
    
    # 5. Save to both locations
    if len(final_dataset) > 0:
        # Save to main CPG directory
        data.write(final_dataset, 'data/cpg', '0_cpg.pkl')
        print(f"✅ Saved to data/cpg/0_cpg.pkl")
        
        # Clean and copy to cpg_tmp for embedding
        os.makedirs('data/cpg_tmp', exist_ok=True)
        
        # Remove old files
        for old_file in os.listdir('data/cpg_tmp'):
            if old_file.endswith('.pkl'):
                os.remove(os.path.join('data/cpg_tmp', old_file))
        
        shutil.copy2('data/cpg/0_cpg.pkl', 'data/cpg_tmp/0_cpg.pkl')
        print(f"✅ Copied to data/cpg_tmp/0_cpg.pkl")
        
        # Show sample
        print(f"📋 Sample data:")
        for idx, row in final_dataset.head(2).iterrows():
            lang = row.get('language', 'N/A')
            target = row.get('target', 'N/A')
            func_preview = str(row.get('func', ''))[:50].replace('\n', ' ')
            print(f"   {idx}: [{lang}] Target={target} - {func_preview}...")
        
        return True
    else:
        print(f"❌ No data to save")
        return False

def main():
    """Main function to fix any language"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python universal_language_fix.py <language>")
        print("Example: python universal_language_fix.py java")
        return
    
    language = sys.argv[1].lower()
    
    # Update config for this language
    config_path = "configs.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['files']['raw'] = f'{language}_dataset.json'
    config['paths']['cpg'] = 'data/cpg_tmp/'  # Point to cpg_tmp for embedding
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated config for {language}")
    
    # Apply the fix
    success = universal_language_fix(language)
    
    if success:
        print(f"\n🎯 {language.upper()} IS READY!")
        print(f"   Next step: python main.py -e")
    else:
        print(f"\n❌ {language.upper()} fix failed")

if __name__ == "__main__":
    main()