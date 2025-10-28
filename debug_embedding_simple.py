#!/usr/bin/env python3
"""
Simple debug script to trace node embedding issues
"""

import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from gensim.models.word2vec import Word2Vec
import json

# Add src to path
sys.path.append('src')
from src.prepare.embeddings import NodesEmbedding, safe_get_node_field
import src.utils.functions.cpg as cpg
import configs

def debug_simple():
    """Simple debug of the embedding process"""
    print("=" * 60)
    print("SIMPLE EMBEDDING DEBUG")
    print("=" * 60)
    
    # Load the Word2Vec model we created
    try:
        w2v_model = Word2Vec.load("data/test_output/test_w2v.model")
        print(f"✓ Loaded Word2Vec model with {len(w2v_model.wv.key_to_index)} words")
        print(f"Vocabulary: {list(w2v_model.wv.key_to_index.keys())}")
    except Exception as e:
        print(f"❌ Error loading Word2Vec: {e}")
        return
    
    # Create a simple test node to debug
    test_node = {
        'id': '1',
        'label': 'void',
        'type': 'METHOD',
        'properties': {
            'code': 'void',
            'name': 'buffer_overflow_test',
            'fullName': 'buffer_overflow_test',
            'signature': 'void buffer_overflow_test(char *input)'
        }
    }
    
    print(f"\n🔍 Testing with sample node:")
    print(f"Node structure: {test_node}")
    
    # Test safe_get_node_field function
    print(f"\n🔍 Testing safe_get_node_field:")
    test_fields = ['code', 'label', 'name', 'type', 'fullName', 'signature']
    
    for field in test_fields:
        try:
            result = safe_get_node_field(test_node, field)
            print(f"  {field}: '{result}' (type: {type(result)})")
        except Exception as e:
            print(f"  {field}: ERROR - {e}")
    
    # Test NodesEmbedding
    print(f"\n🔍 Testing NodesEmbedding:")
    
    try:
        nodes_embedding = NodesEmbedding(100, w2v_model.wv)
        print(f"✓ Created NodesEmbedding")
        
        # Test on single node
        result = nodes_embedding.embed_nodes([test_node])
        print(f"Embedding result:")
        print(f"  Shape: {result.shape}")
        print(f"  Mean: {result.mean().item():.6f}")
        print(f"  Std: {result.std().item():.6f}")
        print(f"  Zero ratio: {(result == 0).float().mean().item():.2%}")
        print(f"  First 5 values: {result[0][:5]}")
        
        if (result == 0).all():
            print(f"  🚨 ALL ZEROS! Let's debug why...")
            
            # Debug step by step
            print(f"\n🔍 Step-by-step debugging:")
            
            # What text is extracted?
            code_text = None
            for field in ('code', 'label', 'name', 'type', 'value'):
                code_text = safe_get_node_field(test_node, field)
                print(f"  Field '{field}': '{code_text}'")
                if code_text and isinstance(code_text, str) and len(code_text.strip()) > 0:
                    print(f"    ✓ Using this field for embedding")
                    break
            
            if not code_text:
                print(f"  🚨 NO TEXT EXTRACTED! This explains the zeros.")
                return
            
            # Tokenize
            import re
            tok_re = re.compile(r'\w+')
            tokens = tok_re.findall(code_text)
            print(f"  Tokens from '{code_text}': {tokens}")
            
            # Check vocabulary
            token_vecs = []
            for t in tokens:
                if t in w2v_model.wv:
                    vec = w2v_model.wv[t]
                    token_vecs.append(vec)
                    print(f"    Token '{t}': FOUND, mean={vec.mean():.3f}")
                else:
                    print(f"    Token '{t}': NOT FOUND in vocabulary")
            
            if not token_vecs:
                print(f"  🚨 NO TOKENS FOUND IN VOCABULARY! This explains the zeros.")
                print(f"  Available vocabulary: {list(w2v_model.wv.key_to_index.keys())}")
            else:
                mean_vec = np.mean(token_vecs, axis=0)
                print(f"  Mean vector: shape={mean_vec.shape}, mean={mean_vec.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Error with NodesEmbedding: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with actual CPG data if available
    print(f"\n🔍 Testing with real CPG data:")
    
    try:
        # Load the mini dataset
        with open("data/dataset_test_mini.json", 'r') as f:
            test_data = json.load(f)
        
        # Create a simple CPG-like structure
        sample_func = test_data[0]['func']
        print(f"Sample function: {sample_func[:100]}...")
        
        # Create mock CPG nodes based on the function
        mock_nodes = [
            {
                'id': '1',
                'label': 'METHOD',
                'type': 'METHOD',
                'properties': {
                    'name': 'buffer_overflow_test',
                    'code': 'void',
                    'fullName': 'buffer_overflow_test'
                }
            },
            {
                'id': '2', 
                'label': 'IDENTIFIER',
                'type': 'IDENTIFIER',
                'properties': {
                    'name': 'buffer',
                    'code': 'buffer'
                }
            },
            {
                'id': '3',
                'label': 'CALL',
                'type': 'CALL', 
                'properties': {
                    'name': 'strcpy',
                    'code': 'strcpy'
                }
            }
        ]
        
        print(f"Testing with mock CPG nodes:")
        for i, node in enumerate(mock_nodes):
            print(f"  Node {i}: {node}")
            
            # Test embedding
            result = nodes_embedding.embed_nodes([node])
            zero_ratio = (result == 0).float().mean().item()
            print(f"    Embedding: mean={result.mean().item():.6f}, zero_ratio={zero_ratio:.2%}")
            
            if zero_ratio > 0.9:
                print(f"    🚨 This node produces mostly zeros!")
            else:
                print(f"    ✓ This node produces valid embeddings")
        
    except Exception as e:
        print(f"❌ Error with real CPG test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple()