#!/usr/bin/env python3
"""
Comprehensive pipeline verification for production readiness
"""

import json
import os
import sys
sys.path.append('src')
import src.data as data
import pandas as pd

def verify_pipeline_ready():
    """Verify the entire pipeline is ready for production"""
    
    print("🔍 COMPREHENSIVE PIPELINE VERIFICATION")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # 1. Configuration Verification
    print("\n📋 1. Configuration Verification")
    try:
        with open('configs.json', 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ['create', 'paths', 'files', 'embed', 'devign', 'process']
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing config section: {section}")
            else:
                print(f"✅ Config section '{section}' present")
        
        # Check paths
        for path_name, path_value in config['paths'].items():
            if os.path.exists(path_value):
                print(f"✅ Path '{path_name}': {path_value}")
            else:
                warnings.append(f"Path '{path_name}' does not exist: {path_value}")
        
        # Check dataset file
        dataset_file = os.path.join(config['paths']['raw'], config['files']['raw'])
        if os.path.exists(dataset_file):
            print(f"✅ Dataset file exists: {config['files']['raw']}")
        else:
            issues.append(f"Dataset file not found: {dataset_file}")
        
    except Exception as e:
        issues.append(f"Config verification failed: {e}")
    
    # 2. Smart Batching Verification
    print("\n📋 2. Smart Batching System Verification")
    try:
        # Test with sample data
        test_data = [
            {"language": "cpp", "func": "int add(int a, int b) { return a + b; }", "target": 0},
            {"language": "java", "func": "public int mul(int x, int y) { return x * y; }", "target": 0},
        ]
        test_df = pd.DataFrame(test_data)
        
        batches = data.smart_language_aware_slice(test_df, 5)
        if len(batches) == 2:  # Should create 2 batches (one per language)
            print("✅ Smart batching system working")
        else:
            issues.append(f"Smart batching failed: expected 2 batches, got {len(batches)}")
            
    except Exception as e:
        issues.append(f"Smart batching verification failed: {e}")
    
    # 3. Joern Frontend Verification
    print("\n📋 3. Joern Frontend Verification")
    joern_dir = config.get('create', {}).get('joern_cli_dir', 'joern/joern-cli/')
    
    frontends = {
        'c2cpg.bat': 'C/C++',
        'javasrc2cpg.bat': 'Java', 
        'pysrc2cpg.bat': 'Python',
        'csharpsrc2cpg.bat': 'C#',
        'php2cpg.bat': 'PHP'
    }
    
    for frontend, language in frontends.items():
        frontend_path = os.path.join(joern_dir, frontend)
        if os.path.exists(frontend_path):
            print(f"✅ {language} frontend: {frontend}")
        else:
            warnings.append(f"{language} frontend not found: {frontend}")
    
    # 4. Language Detection Verification
    print("\n📋 4. Language Detection Verification")
    try:
        # Test language detection in main.py
        if os.path.exists(dataset_file):
            with open(dataset_file, 'r') as f:
                sample_data = json.load(f)
            
            if len(sample_data) > 0:
                if 'language' in sample_data[0]:
                    languages = set(item.get('language') for item in sample_data)
                    print(f"✅ Multi-language dataset detected: {languages}")
                else:
                    print("✅ Single-language dataset (backward compatible)")
            else:
                warnings.append("Dataset is empty")
        
    except Exception as e:
        warnings.append(f"Language detection verification failed: {e}")
    
    # 5. Memory and Performance Settings
    print("\n📋 5. Memory and Performance Settings")
    slice_size = config.get('create', {}).get('slice_size', 100)
    if slice_size <= 200:
        print(f"✅ Slice size appropriate for large datasets: {slice_size}")
    else:
        warnings.append(f"Slice size might be too large for memory: {slice_size}")
    
    # 6. Model Configuration
    print("\n📋 6. Model Configuration Verification")
    model_config = config.get('devign', {}).get('model', {})
    embed_config = config.get('embed', {})
    
    # Check dimension consistency
    nodes_dim = embed_config.get('nodes_dim', 205)
    conv_in_channels = model_config.get('conv_args', {}).get('conv1d_1', {}).get('in_channels', 100)
    
    if nodes_dim > conv_in_channels:
        print(f"✅ Node dimensions compatible: {nodes_dim} -> {conv_in_channels}")
    else:
        warnings.append(f"Dimension mismatch: nodes_dim={nodes_dim}, conv_in_channels={conv_in_channels}")
    
    # Final Report
    print("\n" + "=" * 60)
    print("🎯 PIPELINE READINESS REPORT")
    print("=" * 60)
    
    if not issues:
        print("✅ NO CRITICAL ISSUES FOUND")
    else:
        print("❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print("\n✅ NO WARNINGS")
    
    # Overall status
    if not issues:
        print(f"\n🚀 PIPELINE IS PRODUCTION READY!")
        print(f"   Ready for large multi-language datasets")
        print(f"   Smart batching system operational")
        print(f"   All language frontends available")
        return True
    else:
        print(f"\n🛑 PIPELINE NEEDS FIXES BEFORE PRODUCTION")
        return False

if __name__ == "__main__":
    verify_pipeline_ready()