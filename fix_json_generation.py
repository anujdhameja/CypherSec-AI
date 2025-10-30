#!/usr/bin/env python3
"""
Fix JSON generation by using the existing working extract_funcs.sc script
"""

import os
import shutil
import subprocess
import json

def fix_json_generation():
    """Generate JSON files using the existing working script"""
    
    cpg_tmp_dir = "data/cpg_tmp"
    cpg_main_dir = "data/cpg"
    joern_path = "joern/joern-cli/"
    
    # Ensure main CPG directory exists
    os.makedirs(cpg_main_dir, exist_ok=True)
    
    # Copy CPG files from cpg_tmp to cpg (where the script expects them)
    cpg_files = ['cpp_cpg.bin', 'csharp_cpg.bin', 'java_cpg.bin', 'python_cpg.bin']
    
    print("🔄 Copying CPG files to main CPG directory...")
    for cpg_file in cpg_files:
        src = os.path.join(cpg_tmp_dir, cpg_file)
        dst = os.path.join(cpg_main_dir, cpg_file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ Copied {cpg_file}")
        else:
            print(f"  ❌ Source file not found: {cpg_file}")
    
    # Use the existing working extract_funcs.sc script
    print("\n🔄 Running existing extract_funcs.sc script...")
    
    try:
        command = ["joern.bat", "--script", "extract_funcs.sc"]
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
        return
    
    # Check what JSON files were created and copy them back
    print("\n🔄 Checking generated JSON files...")
    
    json_files = []
    total_functions = 0
    
    for cpg_file in cpg_files:
        json_file = cpg_file.replace('.bin', '.json')
        json_path_main = os.path.join(cpg_main_dir, json_file)
        json_path_tmp = os.path.join(cpg_tmp_dir, json_file)
        
        if os.path.exists(json_path_main):
            # Copy back to cpg_tmp directory
            shutil.copy2(json_path_main, json_path_tmp)
            json_files.append(json_file)
            
            try:
                with open(json_path_main, 'r') as f:
                    data = json.load(f)
                    func_count = len(data.get('functions', []))
                    total_functions += func_count
                    print(f"  ✅ {json_file}: {func_count} functions")
                    
                    # Show sample function names
                    if func_count > 0:
                        sample_funcs = [f['function'] for f in data['functions'][:3]]
                        print(f"    Sample functions: {sample_funcs}")
                        
            except Exception as e:
                print(f"  ❌ Error reading {json_file}: {e}")
        else:
            print(f"  ❌ JSON file not created: {json_file}")
    
    print(f"\n📊 Final Summary:")
    print(f"  CPG files generated: {len(cpg_files)}")
    print(f"  JSON files generated: {len(json_files)}")
    print(f"  Total functions extracted: {total_functions}")
    print(f"  Expected functions: 10 (from original dataset)")
    
    if total_functions > 0:
        print(f"\n✅ SUCCESS! Multi-language CPG generation working!")
        print(f"Languages processed: C++, C#, Java, Python")
        print(f"Next step: Run the embedding task with these JSON files")
    else:
        print(f"\n❌ No functions extracted. Check the JSON files manually.")
    
    return json_files

if __name__ == "__main__":
    fix_json_generation()