#!/usr/bin/env python3
"""
Test the smart language-aware batching system
"""

import json
import sys
sys.path.append('src')
import src.data as data
import pandas as pd

def test_smart_batching():
    """Test the smart batching system with different scenarios"""
    
    print("🧪 TESTING SMART LANGUAGE-AWARE BATCHING")
    print("=" * 60)
    
    # Test 1: Multi-language dataset
    print("\n📋 Test 1: Multi-language dataset")
    
    # Create test dataset
    multilang_data = [
        {"language": "cpp", "func": "int add(int a, int b) { return a + b; }", "target": 0},
        {"language": "cpp", "func": "void copy(char* s) { strcpy(buf, s); }", "target": 1},
        {"language": "cpp", "func": "int mul(int x, int y) { return x * y; }", "target": 0},
        {"language": "java", "func": "public int fibonacci(int n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }", "target": 0},
        {"language": "java", "func": "public void query(String s) { sql = \"SELECT * FROM users WHERE name = '\" + s + \"'\"; }", "target": 1},
        {"language": "python", "func": "def binary_search(arr, target): pass", "target": 0},
        {"language": "python", "func": "def execute(cmd): subprocess.call(cmd, shell=True)", "target": 1},
    ]
    
    df = pd.DataFrame(multilang_data)
    print(f"Input dataset: {len(df)} samples")
    print(f"Languages: {df['language'].unique()}")
    
    # Test with different batch sizes
    for batch_size in [2, 3, 5]:
        print(f"\n🔄 Testing with batch_size={batch_size}")
        batches = data.smart_language_aware_slice(df, batch_size)
        
        print(f"Generated {len(batches)} batches:")
        for batch_id, batch_df, language in batches:
            langs_in_batch = batch_df['language'].unique()
            print(f"  Batch {batch_id}: {len(batch_df)} samples, Language(s): {langs_in_batch}")
            
            # Verify language homogeneity
            if len(langs_in_batch) > 1:
                print(f"    ⚠️  WARNING: Mixed languages in batch!")
            else:
                print(f"    ✅ Homogeneous batch")
    
    # Test 2: Single-language dataset (backward compatibility)
    print(f"\n📋 Test 2: Single-language dataset (backward compatibility)")
    
    single_lang_data = [
        {"func": "int add(int a, int b) { return a + b; }", "target": 0},
        {"func": "void copy(char* s) { strcpy(buf, s); }", "target": 1},
        {"func": "int mul(int x, int y) { return x * y; }", "target": 0},
        {"func": "float div(float a, float b) { return a / b; }", "target": 0},
    ]
    
    df_single = pd.DataFrame(single_lang_data)
    print(f"Input dataset: {len(df_single)} samples (no language column)")
    
    batches_single = data.smart_language_aware_slice(df_single, 2)
    print(f"Generated {len(batches_single)} batches:")
    for batch_id, batch_df, language in batches_single:
        print(f"  Batch {batch_id}: {len(batch_df)} samples, Language: {language}")
    
    # Test 3: Edge cases
    print(f"\n📋 Test 3: Edge cases")
    
    # Empty dataset
    empty_df = pd.DataFrame()
    empty_batches = data.smart_language_aware_slice(empty_df, 5)
    print(f"Empty dataset: {len(empty_batches)} batches")
    
    # Single sample
    single_sample = pd.DataFrame([{"language": "cpp", "func": "int main() { return 0; }", "target": 0}])
    single_batches = data.smart_language_aware_slice(single_sample, 5)
    print(f"Single sample: {len(single_batches)} batches")
    for batch_id, batch_df, language in single_batches:
        print(f"  Batch {batch_id}: {len(batch_df)} samples, Language: {language}")
    
    print(f"\n✅ Smart batching tests completed!")

if __name__ == "__main__":
    test_smart_batching()