#!/usr/bin/env python3
"""
Debug why embeddings are still zero after the fix
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
print("DEBUGGING EMBEDDING ISSUE")
print("="*80)

# Load Word2Vec model
try:
    w2v_model = Word2Vec.load('data/w2v/w2v.model')
    print(f"✓ Loaded Word2Vec model")
    print(f"  Vocabulary size: {len(w2v_model.wv)}")
    print(f"  Sample vocabulary: {list(w2v_model.wv.index_to_key)[:20]}")
except Exception as e:
    print(f"❌ Error loading Word2Vec: {e}")
    exit(1)

# Load a CPG file to get actual nodes
cpg_files = list(Path('data/cpg').glob('*_cpg.pkl'))
if not cpg_files:
    print("❌ No CPG files found")
    exit(1)

print(f"\n📁 Loading CPG file: {cpg_files[0].name}")
cpg_df = pd.read_pickle(cpg_files[0])
print(f"  Samples: {len(cpg_df)}")
print(f"  Columns: {cpg_df.columns.tolist()}")

# Get first sample with nodes
sample_row = None
for idx, row in cpg_df.iterrows():
    if 'nodes' in row and row['nodes'] and len(row['nodes']) > 0:
        sample_row = row
        break

if sample_row is None:
    print("❌ No samples with nodes found")
    exit(1)

print(f"\n🔍 Analyzing sample with {len(sample_row['nodes'])} nodes")

# Test NodesEmbedding directly
nodes_embedding = NodesEmbedding(100, w2v_model.wv)

# Test first few nodes
for i, node in enumerate(sample_row['nodes'][:3]):
    print(f"\n--- Node {i} ---")
    
    # Show node structure
    if isinstance(node, dict):
        print(f"Node keys: {list(node.keys())}")
        if 'properties' in node:
            props = node['properties']
            print(f"Properties keys: {list(props.keys()) if isinstance(props, dict) else 'Not a dict'}")
    else:
        print(f"Node type: {type(node)}")
        if hasattr(node, '__dict__'):
            print(f"Node attributes: {list(node.__dict__.keys())}")
    
    # Test safe_get_node_field
    from src.prepare.embeddings import safe_get_node_field
    
    test_fields = ['code', 'label', 'name', 'type', 'value']
    extracted_text = None
    
    for field in test_fields:
        try:
            value = safe_get_node_field(node, field)
            print(f"  {field}: '{value}' (type: {type(value)})")
            if value and isinstance(value, str) and len(value.strip()) > 0:
                if extracted_text is None:
                    extracted_text = value
        except Exception as e:
            print(f"  {field}: ERROR - {e}")
    
    print(f"  Selected text: '{extracted_text}'")
    
    if extracted_text:
        # Test tokenization
        try:
            tokens = tokenizer(extracted_text)
            print(f"  Tokenized: {tokens}")
            
            # Check vocabulary
            found_tokens = []
            for token in tokens:
                if token in w2v_model.wv:
                    found_tokens.append(token)
                    vec = w2v_model.wv[token]
                    print(f"    '{token}': FOUND, mean={vec.mean():.6f}")
                else:
                    print(f"    '{token}': NOT FOUND")
            
            if not found_tokens:
                print(f"  🚨 NO TOKENS FOUND IN VOCABULARY!")
            
        except Exception as e:
            print(f"  Tokenization error: {e}")
    else:
        print(f"  🚨 NO TEXT EXTRACTED FROM NODE!")

# Test the full embedding process
print(f"\n🔍 Testing full embedding process...")
try:
    result = nodes_embedding.embed_nodes(sample_row['nodes'][:3])
    print(f"Embedding result:")
    print(f"  Shape: {result.shape}")
    print(f"  Mean: {result.mean().item():.6f}")
    print(f"  Std: {result.std().item():.6f}")
    print(f"  Zero ratio: {(result == 0).float().mean().item():.2%}")
    
    if (result == 0).all():
        print(f"  🚨 ALL ZEROS - The fix is not working!")
    else:
        print(f"  ✅ Non-zero embeddings generated")
        
except Exception as e:
    print(f"❌ Embedding error: {e}")
    import traceback
    traceback.print_exc()

print("="*80)