#!/usr/bin/env python3
"""
Inspect the Java PKL file to see what's wrong
"""

import pandas as pd

def inspect_java_pkl():
    """Inspect the Java PKL file"""
    
    pkl_path = "data/cpg/0_cpg.pkl"
    
    print("🔍 INSPECTING JAVA PKL FILE")
    print("=" * 50)
    
    try:
        df = pd.read_pickle(pkl_path)
        
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"\n📋 Sample data:")
            for idx, row in df.iterrows():
                lang = row.get('language', 'N/A')
                target = row.get('target', 'N/A')
                func_preview = str(row.get('func', ''))[:100].replace('\n', ' ')
                
                print(f"  {idx}: [{lang}] Target={target}")
                print(f"      Func: {func_preview}...")
                
                # Check if func is a string (needed for tokenization)
                func_data = row.get('func', '')
                if isinstance(func_data, dict):
                    print(f"      ⚠️  Func is dict, not string: {list(func_data.keys())}")
                elif isinstance(func_data, str):
                    print(f"      ✅ Func is string ({len(func_data)} chars)")
                else:
                    print(f"      ❌ Func is {type(func_data)}")
        else:
            print("❌ PKL file is empty!")
            
    except Exception as e:
        print(f"❌ Error reading PKL file: {e}")

if __name__ == "__main__":
    inspect_java_pkl()