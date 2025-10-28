#!/usr/bin/env python3
"""
Validate that the production embedding fix worked
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path

print("="*80)
print("PRODUCTION EMBEDDING FIX VALIDATION")
print("="*80)

# Load one of the generated input files
input_dir = Path('data/input')
input_files = list(input_dir.glob('*_cpg_input.pkl'))

if not input_files:
    print("❌ No input files found. Run: python main.py -c -e")
    exit(1)

print(f"Found {len(input_files)} input files")

# Test first few files
zero_count = 0
valid_count = 0
total_samples = 0

for file_path in input_files[:5]:  # Test first 5 files
    print(f"\n📁 Testing: {file_path.name}")
    
    try:
        df = pd.read_pickle(file_path)
        print(f"  Samples: {len(df)}")
        
        for idx, row in df.head(3).iterrows():  # Test first 3 samples per file
            graph = row['input']
            total_samples += 1
            
            # Check feature quality
            x = graph.x
            zero_ratio = (x == 0).float().mean().item()
            feature_mean = x.mean().item()
            feature_std = x.std().item()
            
            print(f"    Sample {idx}: nodes={x.shape[0]}, mean={feature_mean:.6f}, std={feature_std:.6f}, zero_ratio={zero_ratio:.2%}")
            
            if zero_ratio > 0.9:
                zero_count += 1
                print(f"      ❌ STILL BROKEN: {zero_ratio:.1%} zeros")
            else:
                valid_count += 1
                print(f"      ✅ FIXED: Valid embeddings")
                
    except Exception as e:
        print(f"  ❌ Error loading {file_path}: {e}")

print(f"\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

success_rate = valid_count / total_samples if total_samples > 0 else 0

print(f"Total samples tested: {total_samples}")
print(f"Valid embeddings: {valid_count}")
print(f"Zero embeddings: {zero_count}")
print(f"Success rate: {success_rate:.1%}")

if success_rate >= 0.9:
    print(f"\n🎉 SUCCESS! The embedding fix is working!")
    print(f"Expected model performance: 60-70%+ accuracy")
    print(f"\n🚀 Ready for next step: Train and test the model")
elif success_rate >= 0.7:
    print(f"\n⚠️  PARTIAL SUCCESS: Most embeddings fixed")
    print(f"Expected model performance: 55-65% accuracy")
else:
    print(f"\n❌ ISSUE: Many embeddings still broken")
    print(f"Need additional debugging")

print("="*80)