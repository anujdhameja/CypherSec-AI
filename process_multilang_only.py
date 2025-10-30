#!/usr/bin/env python3
"""
Process ONLY the multi-language CPG files from cpg_tmp directory
Create JSON -> PKL pipeline for our 4 language CPG files
"""

import os
import subprocess
import json
import pandas as pd
import sys
sys.path.append('src')
import src.data as data
import src.prepare as prepare

def process_multilang_cpg_only():
    """Process only our 4 multi-language CPG files"""
    
    cpg_tmp_dir = "data/cpg_tmp"
    joern_path = "joern/joern-cli/"
    
    # Our specific multi-language CPG files
    multilang_cpg_files = ['cpp_cpg.bin', 'csharp_cpg.bin', 'java_cpg.bin', 'python_cpg.bin']
    
    print("🔄 Processing ONLY multi-language CPG files...")
    
    # Step 1: Create JSON files from our specific CPG files
    print("\n📋 Step 1: Creating JSON files from multi-language CPG files")
    
    # Create a custom script that only processes our specific files
    custom_script = f"""
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import java.io.File

@main def main() = {{
  println("Processing ONLY multi-language CPG files")

  val cpgDir = new File("{os.path.abspath(cpg_tmp_dir).replace(chr(92), chr(92)+chr(92))}")
  
  // Only process our specific files
  val targetFiles = List("cpp_cpg.bin", "csharp_cpg.bin", "java_cpg.bin", "python_cpg.bin")
  
  targetFiles.foreach {{ fileName =>
    val binFile = new File(cpgDir, fileName)
    if (binFile.exists()) {{
      println(s"[*] Processing: $fileName")
      
      val cpg: Cpg = CpgLoader.load(binFile.getAbsolutePath)
      println(s"[*] Successfully loaded CPG from: $fileName")

      val functionsJson = cpg.method.internal.map {{ method =>
        val methodName = if (method.name != null) method.name.replace("\\\\", "\\\\\\\\").replace("\\"", "\\\\\\"") else ""
        val fileName = if (method.location != null && method.location.filename != null) method.location.filename.replace("\\\\", "\\\\\\\\").replace("\\"", "\\\\\\"") else "N/A"

        val nodes = method.ast.map {{ node =>
          val codeStr = if (node.code != null) node.code.replace("\\\\", "\\\\\\\\").replace("\\"", "\\\\\\"").replace("\\n", "\\\\n").replace("\\r", "\\\\r").replace("\\t", "\\\\t") else ""
          val label = if (node.label != null) node.label.replace("\\\\", "\\\\\\\\").replace("\\"", "\\\\\\"") else ""
          s\\"\\"\\"    {{\\"id\\": ${{node.id}}, \\"label\\": \\"$label\\", \\"code\\": \\"$codeStr\\"}}\\"\\"\\"
        }}.l.mkString(",\\n")

        val edges = method.ast.flatMap {{ src =>
          src._astOut.map {{ dst =>
            s\\"\\"\\"    {{\\"source\\": ${{src.id}}, \\"target\\": ${{dst.id}}, \\"label\\": \\"AST\\"}}\\"\\"\\"
          }}
        }}.l.mkString(",\\n")

        s\\"\\"\\"  {{
  \\"function\\": \\"$methodName\\",
  \\"file\\": \\"$fileName\\",
  \\"nodes\\": [
$nodes
  ],
  \\"edges\\": [
$edges
  ]
}}\\"\\"\\"
      }}.l.mkString(",\\n")

      val finalJson = s\\"\\"\\"{{
\\"functions\\": [
$functionsJson
]
}}\\"\\"\\"
      
      val jsonFile = new File(cpgDir, fileName.replace(".bin", ".json"))
      new java.io.PrintWriter(jsonFile) {{ write(finalJson); close() }}
      println(s"[*] Successfully wrote JSON to: ${{jsonFile.getName}}")
    }} else {{
      println(s"[!] File not found: $fileName")
    }}
  }}
}}
"""
    
    # Write the custom script
    script_path = os.path.join(joern_path, "process_multilang.sc")
    with open(script_path, 'w') as f:
        f.write(custom_script)
    
    # Run the script
    try:
        command = ["joern.bat", "--script", "process_multilang.sc"]
        print(f"Running: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            cwd=joern_path,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ JSON generation completed")
        print("Output:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating JSON files: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return
    
    # Step 2: Check generated JSON files
    print("\n📋 Step 2: Checking generated JSON files")
    
    json_files_created = []
    total_functions = 0
    
    for cpg_file in multilang_cpg_files:
        json_file = cpg_file.replace('.bin', '.json')
        json_path = os.path.join(cpg_tmp_dir, json_file)
        
        if os.path.exists(json_path):
            json_files_created.append(json_file)
            
            try:
                with open(json_path, 'r') as f:
                    data_content = json.load(f)
                    func_count = len(data_content.get('functions', []))
                    total_functions += func_count
                    print(f"  ✅ {json_file}: {func_count} functions")
                    
                    # Show function names
                    if func_count > 0:
                        func_names = [f['function'] for f in data_content['functions']]
                        print(f"    Functions: {func_names}")
                        
            except Exception as e:
                print(f"  ❌ Error reading {json_file}: {e}")
        else:
            print(f"  ❌ JSON file not created: {json_file}")
    
    # Step 3: Create PKL files from JSON files
    print(f"\n📋 Step 3: Creating PKL files from JSON files")
    
    # Load our original labeled dataset to get the mapping
    with open('data/raw/dataset_tester_multilang_labeled.json', 'r') as f:
        original_dataset = json.load(f)
    
    original_df = pd.DataFrame(original_dataset)
    
    pkl_files_created = []
    
    for json_file in json_files_created:
        print(f"\n🔄 Processing {json_file}...")
        
        json_path = os.path.join(cpg_tmp_dir, json_file)
        
        # Process JSON to graphs using the existing function
        graphs = prepare.json_process(cpg_tmp_dir, json_file)
        
        if graphs is None or len(graphs) == 0:
            print(f"  ❌ No graphs generated from {json_file}")
            continue
        
        print(f"  ✅ Generated {len(graphs)} graphs from {json_file}")
        
        # Create dataset with CPG
        dataset = data.create_with_index(graphs, ["Index", "cpg"])
        
        # Get the language from filename
        lang = json_file.replace('_cpg.json', '')
        
        # Filter original dataset for this language
        lang_data = original_df[original_df['language'] == lang].copy()
        
        if len(lang_data) == 0:
            print(f"  ❌ No original data found for language: {lang}")
            continue
        
        # Reset index to match with graphs
        lang_data = lang_data.reset_index(drop=True)
        lang_data['Index'] = range(len(lang_data))
        
        print(f"  📊 Language {lang}: {len(lang_data)} original samples, {len(dataset)} CPG samples")
        
        # Join with original data to get func and target
        if len(dataset) <= len(lang_data):
            # Truncate lang_data to match dataset size
            lang_data = lang_data.head(len(dataset))
            lang_data['Index'] = range(len(lang_data))
            
            # Merge
            final_dataset = data.inner_join_by_index(lang_data, dataset)
            
            print(f"  ✅ Created final dataset with {len(final_dataset)} samples")
            print(f"  📋 Columns: {list(final_dataset.columns)}")
            
            # Save PKL file
            pkl_file = f"{lang}_cpg.pkl"
            pkl_path = os.path.join(cpg_tmp_dir, pkl_file)
            
            data.write(final_dataset, cpg_tmp_dir, pkl_file)
            pkl_files_created.append(pkl_file)
            
            print(f"  ✅ Saved PKL file: {pkl_file}")
            
            # Show sample
            if len(final_dataset) > 0:
                sample = final_dataset.iloc[0]
                print(f"  📋 Sample - Target: {sample.get('target', 'N/A')}, Language: {sample.get('language', 'N/A')}")
        else:
            print(f"  ❌ Mismatch: {len(dataset)} CPG samples vs {len(lang_data)} original samples")
    
    # Final summary
    print(f"\n📊 FINAL SUMMARY:")
    print(f"  Multi-language CPG files: {len(multilang_cpg_files)}")
    print(f"  JSON files created: {len(json_files_created)}")
    print(f"  PKL files created: {len(pkl_files_created)}")
    print(f"  Total functions extracted: {total_functions}")
    
    if len(pkl_files_created) > 0:
        print(f"\n✅ SUCCESS! Ready for embedding step (-e)")
        print(f"PKL files in cpg_tmp: {pkl_files_created}")
        print(f"Next: Run 'python main.py -e' to process these PKL files")
    else:
        print(f"\n❌ No PKL files created. Check the process above.")
    
    return pkl_files_created

if __name__ == "__main__":
    process_multilang_cpg_only()