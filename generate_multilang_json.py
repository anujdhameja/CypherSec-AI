#!/usr/bin/env python3
"""
Generate JSON files from the multi-language CPG files
"""

import os
import subprocess
import json

def generate_json_from_cpg():
    """Generate JSON files from CPG files using Joern"""
    
    joern_path = "joern/joern-cli/"
    cpg_path = "data/cpg_tmp/"
    
    # List of CPG files we successfully generated
    cpg_files = ['cpp_cpg.bin', 'java_cpg.bin', 'python_cpg.bin']
    
    json_files = []
    
    for cpg_file in cpg_files:
        print(f"\n🔄 Processing {cpg_file}...")
        
        json_file = cpg_file.replace('.bin', '.json')
        json_path = os.path.join(cpg_path, json_file)
        
        # Use Joern to extract functions from CPG
        script_path = os.path.abspath(os.path.join(joern_path, "extract_funcs.sc"))
        
        try:
            # Set environment variables for Joern
            env = os.environ.copy()
            env['CPG_FILE'] = os.path.abspath(os.path.join(cpg_path, cpg_file))
            env['JSON_FILE'] = os.path.abspath(json_path)
            
            print(f"  CPG file: {env['CPG_FILE']}")
            print(f"  JSON file: {env['JSON_FILE']}")
            
            # Run Joern with the script
            command = ["joern.bat", "--script", script_path]
            print(f"  Running: {' '.join(command)}")
            
            result = subprocess.run(
                command,
                cwd=joern_path,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                env=env
            )
            
            print(f"  ✅ Generated JSON: {json_file}")
            
            # Check if JSON file was created and has content
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    func_count = len(data.get('functions', []))
                    print(f"  📊 Found {func_count} functions in {json_file}")
                    
                json_files.append(json_file)
            else:
                print(f"  ❌ JSON file not created: {json_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error generating JSON for {cpg_file}: {e}")
            print(f"  stdout: {e.stdout}")
            print(f"  stderr: {e.stderr}")
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
    
    print(f"\n✅ Generated {len(json_files)} JSON files: {json_files}")
    
    # Summary of all functions found
    total_functions = 0
    for json_file in json_files:
        json_path = os.path.join(cpg_path, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                func_count = len(data.get('functions', []))
                total_functions += func_count
                print(f"  {json_file}: {func_count} functions")
    
    print(f"\n📊 Total functions extracted: {total_functions}")
    return json_files

if __name__ == "__main__":
    generate_json_from_cpg()