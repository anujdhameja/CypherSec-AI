#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replica Setup Verification
=========================

Quick verification script to ensure all replica components are properly set up
before running the full pipeline test.
"""

import json
import os
import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists


def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status"""
    exists = os.path.exists(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists


def verify_json_dataset(filepath, language):
    """Verify JSON dataset structure"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0:
            print(f"❌ {language.upper()} dataset: Invalid format or empty")
            return False
        
        # Check required fields
        sample = data[0]
        required_fields = ['Sno', 'Primary Language of Benchmark', 'Code Snippet', 'target', 'func']
        
        missing_fields = [field for field in required_fields if field not in sample]
        if missing_fields:
            print(f"❌ {language.upper()} dataset: Missing fields: {missing_fields}")
            return False
        
        print(f"✅ {language.upper()} dataset: {len(data)} samples, format OK")
        return True
        
    except Exception as e:
        print(f"❌ {language.upper()} dataset: Error - {e}")
        return False


def main():
    """Main verification function"""
    print("🔍 REPLICA SETUP VERIFICATION")
    print("="*50)
    
    all_good = True
    
    # 1. Check main configuration files
    print("\n📋 Configuration Files:")
    all_good &= check_file_exists("replica_configs.json", "Replica config")
    all_good &= check_file_exists("configs.json", "Original config")
    
    # 2. Check main scripts
    print("\n🐍 Main Scripts:")
    all_good &= check_file_exists("replica_pipeline_test.py", "Pipeline tester")
    all_good &= check_file_exists("replica_vulnerability_detector.py", "Vulnerability detector")
    all_good &= check_file_exists("run_replica_test.py", "Test runner")
    
    # 3. Check dataset files
    print("\n📊 Dataset Files:")
    languages = ['c', 'cpp', 'csharp', 'python', 'java', 'php']
    
    for lang in languages:
        dataset_path = f"data/replica_raw/replica_dataset_{lang}.json"
        if check_file_exists(dataset_path, f"{lang.upper()} dataset"):
            all_good &= verify_json_dataset(dataset_path, lang)
        else:
            all_good = False
    
    # 4. Check required directories
    print("\n📁 Required Directories:")
    check_directory_exists("data/replica_raw", "Raw data directory")
    
    # Create missing directories (these will be created by the pipeline)
    dirs_to_create = [
        "data/replica_cpg",
        "data/replica_input", 
        "data/replica_tokens",
        "data/replica_w2v",
        "models/replica"
    ]
    
    for dir_path in dirs_to_create:
        if not os.path.exists(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        else:
            print(f"✅ Directory exists: {dir_path}")
    
    # 5. Check Python dependencies
    print("\n🐍 Python Dependencies:")
    required_modules = [
        'torch',
        'torch_geometric', 
        'pandas',
        'numpy',
        'gensim',
        'sklearn'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - Not installed")
            all_good = False
    
    # 6. Check Joern CLI
    print("\n🔧 External Tools:")
    joern_path = "joern/joern-cli/"
    if os.path.exists(joern_path):
        print(f"✅ Joern CLI: {joern_path}")
    else:
        print(f"⚠️  Joern CLI: {joern_path} - Not found (may cause issues)")
    
    # 7. Summary
    print("\n" + "="*50)
    if all_good:
        print("🎉 VERIFICATION PASSED!")
        print("✅ All components are ready")
        print("✅ You can run: python run_replica_test.py")
        print("\n💡 Recommended next steps:")
        print("   1. python run_replica_test.py --quick    # Quick test")
        print("   2. python run_replica_test.py            # Full test")
        print("   3. Review generated reports")
    else:
        print("❌ VERIFICATION FAILED!")
        print("⚠️  Some components are missing or invalid")
        print("🔧 Please fix the issues above before running tests")
        return 1
    
    print("="*50)
    return 0


if __name__ == "__main__":
    sys.exit(main())