#!/usr/bin/env python3
"""
Setup clean testing environment for a specific language
"""

import os
import shutil
import json

def setup_language_test(language):
    """Setup clean testing for a specific language"""
    
    print(f"🔄 Setting up {language.upper()} test environment...")
    
    # 1. Clean cpg_tmp directory
    cpg_tmp_dir = "data/cpg_tmp"
    if os.path.exists(cpg_tmp_dir):
        shutil.rmtree(cpg_tmp_dir)
    os.makedirs(cpg_tmp_dir, exist_ok=True)
    print(f"✅ Cleaned {cpg_tmp_dir}")
    
    # 2. Copy the latest PKL file from cpg to cpg_tmp
    cpg_dir = "data/cpg"
    latest_pkl = os.path.join(cpg_dir, "0_cpg.pkl")
    
    if os.path.exists(latest_pkl):
        target_pkl = os.path.join(cpg_tmp_dir, "0_cpg.pkl")
        shutil.copy2(latest_pkl, target_pkl)
        print(f"✅ Copied latest PKL file to {target_pkl}")
        
        # Verify the PKL file content
        import pandas as pd
        try:
            df = pd.read_pickle(target_pkl)
            if len(df) > 0 and 'language' in df.columns:
                actual_lang = df['language'].iloc[0]
                print(f"📋 PKL file contains {len(df)} samples of language: {actual_lang}")
                
                if actual_lang.lower() != language.lower():
                    print(f"⚠️  WARNING: Expected {language} but PKL contains {actual_lang}")
                else:
                    print(f"✅ Language match confirmed: {actual_lang}")
            else:
                print(f"❌ PKL file appears to be empty or invalid")
        except Exception as e:
            print(f"❌ Error reading PKL file: {e}")
    else:
        print(f"❌ Latest PKL file not found: {latest_pkl}")
        return False
    
    # 3. Update config to use cpg_tmp and the language dataset
    config_path = "configs.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update paths and files
        config['paths']['cpg'] = 'data/cpg_tmp/'
        config['files']['raw'] = f'{language}_dataset.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Updated config:")
        print(f"   - CPG path: data/cpg_tmp/")
        print(f"   - Raw file: {language}_dataset.json")
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False
    
    print(f"\n🎯 {language.upper()} test environment ready!")
    print(f"   Next step: python main.py -e")
    
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python setup_language_test.py <language>")
        print("Example: python setup_language_test.py java")
        sys.exit(1)
    
    language = sys.argv[1].lower()
    setup_language_test(language)