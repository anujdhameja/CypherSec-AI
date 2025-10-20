"""
Inspect the data files to understand the structure
"""

import pandas as pd
import torch
import pickle

print("="*80)
print("INSPECTING DATA FILES")
print("="*80)

# Load first data file
file_path = 'data/input/0_cpg_input.pkl'
print(f"\nLoading: {file_path}")

try:
    df = pd.read_pickle(file_path)
    print(f"✓ Loaded successfully")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame dtypes:\n{df.dtypes}")
    
    # Check first few rows
    print(f"\nFirst 3 rows:")
    for i in range(min(3, len(df))):
        print(f"\nRow {i}:")
        for col in df.columns:
            val = df.iloc[i][col]
            print(f"  {col}: {type(val)} - {str(val)[:100]}...")
            
            # If it's a PyG Data object, inspect it
            if hasattr(val, 'x'):
                print(f"    x shape: {val.x.shape}")
                print(f"    x sample: {val.x[:3, :5]}")  # First 3 nodes, first 5 features
                print(f"    x range: [{val.x.min():.6f}, {val.x.max():.6f}]")
                print(f"    x mean: {val.x.mean():.6f}")
                print(f"    x std: {val.x.std():.6f}")
                print(f"    x nonzero: {(val.x != 0).sum().item()}/{val.x.numel()}")
            
            if hasattr(val, 'edge_index'):
                print(f"    edge_index shape: {val.edge_index.shape}")
                print(f"    edge_index sample: {val.edge_index[:, :5]}")
            
            if hasattr(val, 'y'):
                print(f"    y: {val.y}")

except Exception as e:
    print(f"❌ Error loading file: {e}")
    import traceback
    traceback.print_exc()

# Also check if there are any Word2Vec models
print(f"\n" + "="*50)
print("CHECKING WORD2VEC MODEL")
print("="*50)

import os
w2v_path = 'data/w2v/'
if os.path.exists(w2v_path):
    files = os.listdir(w2v_path)
    print(f"W2V directory contents: {files}")
    
    # Try to load w2v model
    w2v_file = 'data/w2v/w2v.model'
    if os.path.exists(w2v_file):
        try:
            from gensim.models import Word2Vec
            w2v_model = Word2Vec.load(w2v_file)
            print(f"✓ W2V model loaded")
            print(f"Vocabulary size: {len(w2v_model.wv)}")
            print(f"Vector size: {w2v_model.wv.vector_size}")
            
            # Test a few words
            vocab_sample = list(w2v_model.wv.key_to_index.keys())[:10]
            print(f"Sample vocabulary: {vocab_sample}")
            
            if vocab_sample:
                test_word = vocab_sample[0]
                vector = w2v_model.wv[test_word]
                print(f"Sample vector for '{test_word}': {vector[:5]}...")
                print(f"Vector range: [{vector.min():.6f}, {vector.max():.6f}]")
                
        except Exception as e:
            print(f"❌ Error loading W2V model: {e}")
    else:
        print(f"❌ W2V model file not found: {w2v_file}")
else:
    print(f"❌ W2V directory not found: {w2v_path}")

print(f"\n" + "="*80)
print("INSPECTION COMPLETE")
print("="*80)