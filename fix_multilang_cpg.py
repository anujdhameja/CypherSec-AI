#!/usr/bin/env python3
"""
Fix for multi-language CPG generation
This script modifies the pipeline to handle multiple languages properly
"""

import os
import shutil
import subprocess
import pandas as pd
from pathlib import Path

def create_multilang_cpg():
    """Create CPG files for multi-language dataset by grouping by language"""
    
    # Load the labeled dataset
    import json
    with open('data/raw/dataset_tester_multilang_labeled.json', 'r') as f:
        dataset = json.load(f)
    
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} samples with languages: {df['language'].unique()}")
    
    # Group by language
    language_groups = df.groupby('language')
    
    cpg_files = []
    joern_path = "joern/joern-cli/"
    
    for lang, group in language_groups:
        print(f"\n🔄 Processing {len(group)} {lang} functions...")
        
        # Create temporary directory for this language
        lang_dir = f"data/joern_{lang}"
        os.makedirs(lang_dir, exist_ok=True)
        
        # Create files for this language
        ext_map = {
            'c': '.c',
            'cpp': '.cpp', 
            'csharp': '.cs',
            'python': '.py',
            'java': '.java',
            'php': '.php'
        }
        
        file_ext = ext_map.get(lang, '.c')
        
        for idx, row in group.iterrows():
            file_name = f"{idx}{file_ext}"
            file_path = os.path.join(lang_dir, file_name)
            
            with open(file_path, 'w') as f:
                f.write(row['func'])
            
            print(f"  Created: {file_name}")
        
        # Generate CPG for this language
        cpg_file = f"{lang}_cpg.bin"
        abs_input_path = os.path.abspath(lang_dir)
        abs_output_path = os.path.abspath(os.path.join("data/cpg_tmp", cpg_file))
        
        # Use appropriate Joern frontend based on language
        frontend_map = {
            'c': 'c2cpg.bat',
            'cpp': 'c2cpg.bat',
            'csharp': 'csharp2cpg.bat', 
            'python': 'pysrc2cpg.bat',
            'java': 'javasrc2cpg.bat',
            'php': 'php2cpg.bat'
        }
        
        frontend = frontend_map.get(lang, 'joern-parse.bat')
        
        try:
            if frontend == 'joern-parse.bat':
                # Generic parser
                command = [frontend, abs_input_path, "--output", abs_output_path]
            else:
                # Language-specific parser
                command = [frontend, abs_input_path, "-o", abs_output_path]
            
            print(f"  Running: {' '.join(command)}")
            result = subprocess.run(command, cwd=joern_path, shell=True, check=True, 
                                  capture_output=True, text=True)
            
            print(f"  ✅ Generated CPG: {cpg_file}")
            cpg_files.append(cpg_file)
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error generating CPG for {lang}: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
        
        # Clean up temporary directory
        shutil.rmtree(lang_dir)
    
    print(f"\n✅ Generated {len(cpg_files)} CPG files: {cpg_files}")
    return cpg_files

if __name__ == "__main__":
    create_multilang_cpg()