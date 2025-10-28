"""
Validate the embedding fix using existing production data
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')
from src.prepare.embeddings import NodesEmbedding
from src.utils.functions.parse import tokenizer
from gensim.models.word2vec import Word2Vec

print("="*80)
print("PRODUCTION EMBEDDING FIX VALIDATION")
print("="*80)

# Test the tokenizer function directly
print("\n1. TESTING TOKENIZER FUNCTION")
print("="*40)

test_cases = [
    'void',
    'buffer', 
    'strcpy',
    'buffer_overflow_test',
    'malloc',
    'free'
]

print("Testing tokenizer on individual words:")
for word in test_cases:
    tokens = tokenizer(word)
    print(f"  '{word}' → {tokens}")

# Load existing Word2Vec model if available
print("\n2. TESTING WITH WORD2VEC MODEL")
print("="*40)

w2v_paths = [
    'data/test_output/test_w2v.model',
    'data/w2v/w2v.model',
    'data/w2v/w2v'
]

w2v_model = None
for path in w2v_paths:
    try:
        w2v_model = Word2Vec.load(path)
        print(f"✓ Loaded Word2Vec from: {path}")
        print(f"  Vocabulary size: {len(w2v_model.wv)}")
        print(f"  Sample vocabulary: {list(w2v_model.wv.index_to_key)[:10]}")
        break
    except Exception as e:
        print(f"  Could not load {path}: {e}")

if w2v_model is None:
    print("❌ No Word2Vec model found. Cannot test embeddings.")
    exit(1)

# Test NodesEmbedding with the fix
print("\n3. TESTING NODESEMBEDDING CLASS")
print("="*40)

nodes_embedding = NodesEmbedding(100, w2v_model.wv)

test_nodes = [
    {'id': '1', 'properties': {'code': 'void'}},
    {'id': '2', 'properties': {'code': 'buffer'}},
    {'id': '3', 'properties': {'code': 'strcpy'}},
    {'id': '4', 'properties': {'code': 'buffer_overflow_test'}},
    {'id': '5', 'properties': {'code': 'malloc'}},
]

print("Testing node embeddings:")
success_count = 0
total_count = len(test_nodes)

for node in test_nodes:
    code = node['properties']['code']
    
    # Show tokenization
    tokens = tokenizer(code)
    print(f"\n  Node '{code}':")
    print(f"    Tokenized: {tokens}")
    
    # Check vocabulary coverage
    found_tokens = []
    for token in tokens:
        if token in w2v_model.wv:
            found_tokens.append(token)
    
    print(f"    Found in vocab: {found_tokens}")
    
    # Test embedding
    try:
        result = nodes_embedding.embed_nodes([node])
        zero_ratio = (result == 0).float().mean().item()
        
        print(f"    Embedding shape: {result.shape}")
        print(f"    Mean: {result.mean().item():.6f}")
        print(f"    Zero ratio: {zero_ratio:.2%}")
        
        if zero_ratio < 0.1:
            print(f"    ✅ SUCCESS: Valid embedding")
            success_count += 1
        else:
            print(f"    ❌ FAILED: Still mostly zeros")
            
    except Exception as e:
        print(f"    ❌ ERROR: {e}")

print(f"\n4. SUMMARY")
print("="*40)
print(f"Successful embeddings: {success_count}/{total_count}")
print(f"Success rate: {success_count/total_count*100:.1f}%")

if success_count == total_count:
    print("\n✅ EMBEDDING FIX IS WORKING!")
    print("All test nodes produce valid embeddings.")
    print("The tokenization mismatch has been resolved.")
elif success_count >= total_count * 0.8:
    print("\n⚠️  PARTIAL SUCCESS")
    print("Most nodes work, but some issues remain.")
else:
    print("\n❌ FIX NOT WORKING")
    print("Most nodes still produce zero embeddings.")
    print("Additional debugging needed.")

print("\n" + "="*80)