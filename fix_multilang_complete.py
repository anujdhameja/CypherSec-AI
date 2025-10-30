#!/usr/bin/env python3
"""
Complete fix for multi-language CPG generation and JSON extraction
"""

import os
import shutil
import subprocess
import pandas as pd
import json
from pathlib import Path

def create_multilang_cpg_complete():
    """Create CPG files and JSON for multi-language dataset"""
    
    # Load the labeled dataset
    with open('data/raw/dataset_tester_multilang_labeled.json', 'r') as f:
        dataset = json.load(f)
    
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} samples with languages: {df['language'].unique()}")
    
    # Group by language
    language_groups = df.groupby('language')
    
    cpg_files = []
    joern_path = "joern/joern-cli/"
    cpg_dir = "data/cpg_tmp"
    
    # Ensure CPG directory exists
    os.makedirs(cpg_dir, exist_ok=True)
    
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
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(row['func'])
            
            print(f"  Created: {file_name}")
        
        # Generate CPG for this language
        cpg_file = f"{lang}_cpg.bin"
        abs_input_path = os.path.abspath(lang_dir)
        abs_output_path = os.path.abspath(os.path.join(cpg_dir, cpg_file))
        
        # Use correct frontend names
        frontend_map = {
            'c': 'c2cpg.bat',
            'cpp': 'c2cpg.bat',
            'csharp': 'csharpsrc2cpg.bat',  # Fixed name
            'python': 'pysrc2cpg.bat',
            'java': 'javasrc2cpg.bat',
            'php': 'php2cpg.bat'
        }
        
        frontend = frontend_map.get(lang, 'joern-parse.bat')
        
        try:
            if lang == 'php':
                # Skip PHP for now since PHP runtime is not installed
                print(f"  ⚠️  Skipping {lang} - PHP runtime not installed")
                continue
            
            if frontend == 'joern-parse.bat':
                # Generic parser
                command = [frontend, abs_input_path, "--output", abs_output_path]
            else:
                # Language-specific parser
                command = [frontend, abs_input_path, "-o", abs_output_path]
            
            print(f"  Running: {' '.join(command)}")
            result = subprocess.run(command, cwd=joern_path, shell=True, check=True, 
                                  capture_output=True, text=True)
            
            if os.path.exists(abs_output_path):
                print(f"  ✅ Generated CPG: {cpg_file}")
                cpg_files.append(cpg_file)
            else:
                print(f"  ❌ CPG file not created: {cpg_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error generating CPG for {lang}: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
        
        # Clean up temporary directory
        shutil.rmtree(lang_dir)
    
    print(f"\n✅ Generated {len(cpg_files)} CPG files: {cpg_files}")
    
    # Now generate JSON files from CPG files
    print(f"\n🔄 Generating JSON files from CPG files...")
    
    # Create a custom Joern script for our CPG directory
    custom_script = f"""
import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import java.io.File

@main def main() = {{
  println("Executing custom extract_funcs.sc")

  val inDir = new File("{os.path.abspath(cpg_dir).replace(chr(92), chr(92)+chr(92))}")
  val outD  = new File("{os.path.abspath(cpg_dir).replace(chr(92), chr(92)+chr(92))}")
  outD.mkdirs()

  val binFiles = inDir.listFiles().filter(_.getName.endsWith(".bin"))
  println(s"[*] Found ${{binFiles.size}} CPG bin files.")

  // Helper: robust JSON string escaping
  def escape(s: String): String = {{
    if (s == null) ""
    else s.flatMap {{
      case '\\\\' => "\\\\\\\\"
      case '\"' => "\\\\\""
      case '\\n' => "\\\\n"
      case '\\r' => "\\\\r"
      case '\\t' => "\\\\t"
      case c if c.isControl => ""  // remove other control chars
      case c => c.toString
    }}
  }}

  binFiles.foreach {{ file =>
    println(s"[*] Loading CPG from: ${{file.getName}}")
    val cpg: Cpg = CpgLoader.load(file.getAbsolutePath)
    println(s"[*] Successfully loaded CPG from: ${{file.getName}}")

    val functionsJson = cpg.method.internal.map {{ method =>
      val methodName = escape(method.name)
      val fileName = if (method.location != null && method.location.filename != null) escape(method.location.filename) else "N/A"

      val nodes = method.ast.map {{ node =>
        val codeStr = escape(node.code)
        s\"\"\"    {{"id": ${{node.id}}, "label": "${{escape(node.label)}}", "code": "$codeStr"}}\"\"\"
      }}.l.mkString(",\\n")

      val edges = method.ast.flatMap {{ src =>
        src._astOut.map {{ dst =>
          s\"\"\"    {{"source": ${{src.id}}, "target": ${{dst.id}}, "label": "AST"}}\"\"\"
        }}
      }}.l.mkString(",\\n")

      s\"\"\"  {{
  "function": "$methodName",
  "file": "$fileName",
  "nodes": [
$nodes
  ],
  "edges": [
$edges
  ]
}}\"\"\"
    }}.l.mkString(",\\n")

    val finalJson = s\"\"\"{{
"functions": [
$functionsJson
]
}}\"\"\"
    val outFile = new File(outD, file.getName.replace(".bin", ".json"))
    new java.io.PrintWriter(outFile) {{ write(finalJson); close() }}
    println(s"[*] Successfully wrote graph data to ${{outFile.getName}}")
  }}
}}
"""
    
    # Write custom script
    custom_script_path = os.path.join(joern_path, "extract_funcs_custom.sc")
    with open(custom_script_path, 'w') as f:
        f.write(custom_script)
    
    # Run the custom script
    try:
        command = ["joern.bat", "--script", "extract_funcs_custom.sc"]
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
        print("stdout:", result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating JSON files: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
    
    # Check what JSON files were created
    json_files = []
    total_functions = 0
    
    for cpg_file in cpg_files:
        json_file = cpg_file.replace('.bin', '.json')
        json_path = os.path.join(cpg_dir, json_file)
        
        if os.path.exists(json_path):
            json_files.append(json_file)
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    func_count = len(data.get('functions', []))
                    total_functions += func_count
                    print(f"  ✅ {json_file}: {func_count} functions")
            except Exception as e:
                print(f"  ❌ Error reading {json_file}: {e}")
        else:
            print(f"  ❌ JSON file not created: {json_file}")
    
    print(f"\n📊 Summary:")
    print(f"  CPG files: {len(cpg_files)}")
    print(f"  JSON files: {len(json_files)}")
    print(f"  Total functions: {total_functions}")
    print(f"  Expected functions: {len(df)}")
    
    return cpg_files, json_files

if __name__ == "__main__":
    create_multilang_cpg_complete()