"""
Validate that the tokenization fix is working correctly.
Compare embedding quality before and after the fix.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from gensim.models import Word2Vec

print("="*80)
print("EMBEDDING FIX VALIDATION")
print("="*80)

# Load the test data that was created with the fix
test_input_path = Path('data/test_output/test_input.pkl')
test_w2v_path = Path('data/test_output/test_w2v.model')
test_tokens_path = Path('data/test_output/test_tokens.pkl')

if not test_input_path.exists():
    print("❌ Test output not found. Run: python main.py -t")
    exit(1)

print("\n" + "="*80)
print("1. LOADING TEST DATA")
print("="*80)

# Load the generated data
try:
    test_data = pd.read_pickle(test_input_path)
    print(f"✓ Loaded test input data: {len(test_data)} samples")
    print(f"  Columns: {test_data.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading test data: {e}")
    exit(1)

try:
    w2v_model = Word2Vec.load(str(test_w2v_path))
    print(f"✓ Loaded Word2Vec model: {len(w2v_model.wv)} vocabulary")
    print(f"  Vector size: {w2v_model.vector_size}")
except Exception as e:
    print(f"❌ Error loading Word2Vec: {e}")
    exit(1)

try:
    tokens_df = pd.read_pickle(test_tokens_path)
    print(f"✓ Loaded tokens: {len(tokens_df)} samples")
except Exception as e:
    print(f"❌ Error loading tokens: {e}")
    exit(1)

print("\n" + "="*80)
print("2. ANALYZING FEATURE QUALITY")
print("="*80)

# Analyze the generated input tensors
zero_features = 0
non_zero_features = 0
feature_stats = []

for idx, row in test_data.iterrows():
    if 'input' in row:
        graph = row['input']
        if hasattr(graph, 'x'):
            x = graph.x
            zero_ratio = (x == 0).all(dim=1).sum().item() / x.shape[0]
            feature_mean = x.mean().item()
            feature_std = x.std().item()
            
            if zero_ratio > 0.9:  # More than 90% zero
                zero_features += 1
            else:
                non_zero_features += 1
                
            feature_stats.append({
                'index': idx,
                'zero_ratio': zero_ratio,
                'mean': feature_mean,
                'std': feature_std,
                'nodes': x.shape[0]
            })

print(f"\n📊 Feature Quality Statistics:")
print(f"  Non-zero feature graphs: {non_zero_features}/{len(test_data)}")
print(f"  Zero feature graphs: {zero_features}/{len(test_data)}")
print(f"  Success rate: {non_zero_features/len(test_data)*100:.1f}%")

if non_zero_features > 0:
    feature_df = pd.DataFrame(feature_stats)
    print(f"\n  Average feature mean: {feature_df['mean'].mean():.6f}")
    print(f"  Average feature std: {feature_df['std'].mean():.6f}")
    print(f"  Average zero ratio: {feature_df['zero_ratio'].mean()*100:.2f}%")

print("\n" + "="*80)
print("3. COMPARING WITH TRAINING TOKENS")
print("="*80)

# Check vocabulary coverage
all_tokens = []
for idx, row in tokens_df.iterrows():
    if 'tokens' in row:
        all_tokens.extend(row['tokens'])

unique_tokens = set(all_tokens)
in_vocab = sum(1 for t in unique_tokens if t in w2v_model.wv)
total_unique = len(unique_tokens)

print(f"\n📚 Vocabulary Coverage:")
print(f"  Unique tokens in data: {total_unique}")
print(f"  Tokens in W2V vocab: {in_vocab}")
print(f"  Coverage rate: {in_vocab/total_unique*100:.1f}%")
print(f"  Missing tokens: {total_unique - in_vocab}")

if in_vocab < total_unique:
    missing = unique_tokens - set(w2v_model.wv.index_to_key)
    print(f"\n  Missing from vocabulary: {list(missing)[:10]}")

print("\n" + "="*80)
print("4. SAMPLE ANALYSIS")
print("="*80)

# Show detailed analysis of first few samples
for idx in range(min(3, len(test_data))):
    row = test_data.iloc[idx]
    if 'input' in row:
        graph = row['input']
        print(f"\n📊 Sample {idx}:")
        print(f"  Label: {graph.y.item()}")
        print(f"  Nodes: {graph.x.shape[0]}")
        print(f"  Feature shape: {graph.x.shape}")
        
        # Check feature quality
        zero_count = (graph.x == 0).all(dim=1).sum().item()
        non_zero_count = graph.x.shape[0] - zero_count
        print(f"  Non-zero nodes: {non_zero_count}/{graph.x.shape[0]}")
        print(f"  Feature mean: {graph.x.mean():.6f}")
        print(f"  Feature std: {graph.x.std():.6f}")
        print(f"  Feature range: [{graph.x.min():.4f}, {graph.x.max():.4f}]")
        
        if zero_count > 0:
            print(f"  ⚠️  {zero_count} nodes with zero features")
        else:
            print(f"  ✅ All nodes have valid features")

print("\n" + "="*80)
print("5. FINAL VERDICT")
print("="*80)

if non_zero_features == len(test_data):
    print("✅ SUCCESS! The embedding fix is working!")
    print(f"   All {len(test_data)} samples have valid (non-zero) features.")
    print(f"   Expected model performance: 60-70%+ accuracy")
    print(f"\n🚀 Next steps:")
    print(f"   1. Re-run full pipeline: python main.py")
    print(f"   2. Train model: python auto_hyperparameter_FIXED.py")
    print(f"   3. Validate performance: python compare_models_fairly.py")
elif non_zero_features >= len(test_data) * 0.9:
    print("⚠️  PARTIAL SUCCESS - Most features are fixed")
    print(f"   {non_zero_features}/{len(test_data)} samples have valid features")
    print(f"   {zero_features} samples still have issues")
    print(f"   Expected model performance: 55-65% accuracy")
else:
    print("❌ ISSUE REMAINS - Most features are still zero")
    print(f"   {non_zero_features}/{len(test_data)} samples have valid features")
    print(f"   {zero_features} samples have zero features")
    print(f"   The tokenization fix may not be complete")

print("\n" + "="*80)
print("Analysis complete. Check results above.")
print("="*80)