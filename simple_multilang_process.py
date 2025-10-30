#!/usr/bin/env python3
"""
Simple approach: Process each multi-language CPG file individually
"""

import os
import subprocess
import json
import pandas as pd
import sys
sys.path.append('src')
import src.data as data
import src.prepare as prepare

def process_single_cpg_to_json(cpg_file, cpg_tmp_dir, joern_path):
    """Process a single CPG file to JSON using individual Joern command"""
    
    cpg_path = os.path.join(cpg_tmp_dir, cpg_file)
    json_file = cpg_file.replace('.bin', '.json')
    json_path = os.path.join(cpg_tmp_dir, json_file)
    
    print(f"🔄 Processing {cpg_file} -> {json_file}")
    
    # Create a simple script for this specific file
    simple_script = f'''
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import java.io.File

@main def main() = {{
  val cpgFile = new File("{os.path.abspath(cpg_path).replace(chr(92), chr(92)+chr(92))}")
  val jsonFile = new File("{os.path.abspath(json_path).replace(chr(92), chr(92)+chr(92))}")
  
  println(s"Loading CPG from: ${{cpgFile.getName}}")
  val cpg: Cpg = CpgLoader.load(cpgFile.getAbsolutePath)
  
  val functions = cpg.method.internal.l
  println(s"Found ${{functions.size}} methods")
  
  val functionsJson = functions.map {{ method =>
    val methodName = if (method.name != null) method.name else ""
    val fileName = if (method.location != null && method.location.filename != null) method.location.filename else "N/A"
    
    val nodes = method.ast.l.map {{ node =>
      val nodeId = node.id
      val label = if (node.label != null) node.label else ""
      val code = if (node.code != null) node.code else ""
      
      // Simple JSON without complex escaping
      s"""    {{"id": $nodeId, "label": "$label", "code": "$code"}}"""
    }}.mkString(",\\n")
    
    val edges = method.ast.l.flatMap {{ src =>
      src._astOut.l.map {{ dst =>
        s"""    {{"source": ${{src.id}}, "target": ${{dst.id}}, "label": "AST"}}"""
      }}
    }}.mkString(",\\n")
    
    s"""  {{
  "function": "$methodName",
  "file": "$fileName", 
  "nodes": [
$nodes
  ],
  "edges": [
$edges
  ]
}}"""
  }}.mkString(",\\n")
  
  val finalJson = s"""{{
"functions": [
$functionsJson
]
}}"""
  
  new java.io.PrintWriter(jsonFile) {{ write(finalJson); close() }}
  println(s"Successfully wrote JSON to: ${{jsonFile.getName}}")
}}
'''
    
    # Write script
    script_name = f"process_{cpg_file.replace('.bin', '')}.sc"
    script_path = os.path.join(joern_path, script_name)
    
    with open(script_path, 'w') as f:
        f.write(simple_script)
    
    # Run script
    try:
        command = ["joern.bat", "--script", script_name]
        result = subprocess.run(
            command,
            cwd=joern_path,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"  ✅ Generated {json_file}")
        print(f"  Output: {result.stdout.strip()}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: {e}")
        print(f"  stderr: {e.stderr}")
        return False

def simple_multilang_process():
    """Simple processing of multi-language CPG files"""
    
    cpg_tmp_dir = "data/cpg_tmp"
    joern_path = "joern/joern-cli/"
    
    # Our specific multi-language CPG files
    multilang_cpg_files = ['cpp_cpg.bin', 'csharp_cpg.bin', 'java_cpg.bin', 'python_cpg.bin']
    
    print("🔄 Simple multi-language CPG processing...")
    
    # Step 1: Process each CPG file to JSON
    print("\n📋 Step 1: Converting CPG files to JSON")
    
    json_files_created = []
    
    for cpg_file in multilang_cpg_files:
        cpg_path = os.path.join(cpg_tmp_dir, cpg_file)
        
        if os.path.exists(cpg_path):
            success = process_single_cpg_to_json(cpg_file, cpg_tmp_dir, joern_path)
            if success:
                json_files_created.append(cpg_file.replace('.bin', '.json'))
        else:
            print(f"  ❌ CPG file not found: {cpg_file}")
    
    # Step 2: Check JSON files
    print(f"\n📋 Step 2: Checking JSON files")
    
    total_functions = 0
    valid_json_files = []
    
    for json_file in json_files_created:
        json_path = os.path.join(cpg_tmp_dir, json_file)
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data_content = json.load(f)
                    func_count = len(data_content.get('functions', []))
                    total_functions += func_count
                    valid_json_files.append(json_file)
                    
                    print(f"  ✅ {json_file}: {func_count} functions")
                    
                    # Show function names
                    if func_count > 0:
                        func_names = [f['function'] for f in data_content['functions'][:3]]
                        print(f"    Sample functions: {func_names}")
                        
            except Exception as e:
                print(f"  ❌ Error reading {json_file}: {e}")
        else:
            print(f"  ❌ JSON file not found: {json_file}")
    
    # Step 3: Create PKL files
    print(f"\n📋 Step 3: Creating PKL files")
    
    # Load original dataset
    with open('data/raw/dataset_tester_multilang_labeled.json', 'r') as f:
        original_dataset = json.load(f)
    
    original_df = pd.DataFrame(original_dataset)
    
    pkl_files_created = []
    
    for json_file in valid_json_files:
        print(f"\n🔄 Processing {json_file} to PKL...")
        
        # Get language from filename
        lang = json_file.replace('_cpg.json', '')
        
        # Process JSON to graphs
        graphs = prepare.json_process(cpg_tmp_dir, json_file)
        
        if graphs is None or len(graphs) == 0:
            print(f"  ❌ No graphs from {json_file}")
            continue
        
        print(f"  ✅ Generated {len(graphs)} graphs")
        
        # Create dataset with CPG
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        
        # Get original data for this language
        lang_data = original_df[original_df['language'] == lang].copy().reset_index(drop=True)
        lang_data['Index'] = range(len(lang_data))
        
        print(f"  📊 {lang}: {len(lang_data)} original, {len(dataset)} CPG")
        
        # Match sizes and merge
        min_size = min(len(dataset), len(lang_data))
        if min_size > 0:
            dataset_trimmed = dataset.head(min_size)
            lang_data_trimmed = lang_data.head(min_size)
            lang_data_trimmed['Index'] = range(min_size)
            
            # Merge
            final_dataset = data.inner_join_by_index(lang_data_trimmed, dataset_trimmed)
            
            print(f"  ✅ Final dataset: {len(final_dataset)} samples")
            print(f"  📋 Columns: {list(final_dataset.columns)}")
            
            # Save PKL
            pkl_file = f"{lang}_cpg.pkl"
            data.write(final_dataset, cpg_tmp_dir, pkl_file)
            pkl_files_created.append(pkl_file)
            
            print(f"  ✅ Saved: {pkl_file}")
            
            # Show sample
            if len(final_dataset) > 0:
                sample = final_dataset.iloc[0]
                print(f"  📋 Sample - Target: {sample.get('target')}, Lang: {sample.get('language')}")
        else:
            print(f"  ❌ No data to merge for {lang}")
    
    # Final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  CPG files processed: {len(multilang_cpg_files)}")
    print(f"  JSON files created: {len(valid_json_files)}")
    print(f"  PKL files created: {len(pkl_files_created)}")
    print(f"  Total functions: {total_functions}")
    
    if len(pkl_files_created) > 0:
        print(f"\n✅ SUCCESS! Ready for embedding step")
        print(f"PKL files: {pkl_files_created}")
        print(f"Next: python main.py -e")
    else:
        print(f"\n❌ No PKL files created")
    
    return pkl_files_created

if __name__ == "__main__":
    simple_multilang_process()