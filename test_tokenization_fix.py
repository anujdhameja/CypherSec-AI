#!/usr/bin/env python3
"""
Test the tokenization fix
"""

import sys
sys.path.append('src')

from src.utils.functions.parse import tokenizer
from src.prepare.embeddings import NodesEmbedding
from gensim.models.word2vec import Word2Vec
import numpy as np

def test_tokenization_fix():
    print("=" * 60)
    print("TESTING TOKENIZATION FIX")
    print("=" * 60)
    
    # Test the tokenizer function
    test_texts = [
        'buffer_overflow_test',
        'buffer',
        'strcpy',
        'void',
        'char'
    ]
    
    print("\n🔍 Testing tokenizer function:")
    for text in test_texts:
        tokens = tokenizer(text)
        print(f"  '{text}' → {tokens}")
    
    # Load the Word2Vec model
    try:
        w2v_model = Word2Vec.load("data/test_output/test_w2v.model")
        print(f"\n✓ Loaded Word2Vec model with vocabulary: {list(w2v_model.wv.key_to_index.keys())}")
    except Exception as e:
        print(f"❌ Error loading Word2Vec: {e}")
        return
    
    # Test the fixed NodesEmbedding
    print(f"\n🔍 Testing fixed NodesEmbedding:")
    
    test_nodes = [
        {
            'id': '1',
            'label': 'buffer_overflow_test',
            'properties': {'code': 'buffer_overflow_test'}
        },
        {
            'id': '2',
            'label': 'buffer',
            'properties': {'code': 'buffer'}
        },
        {
            'id': '3',
            'label': 'strcpy',
            'properties': {'code': 'strcpy'}
        },
        {
            'id': '4',
            'label': 'void',
            'properties': {'code': 'void'}
        }
    ]
    
    nodes_embedding = NodesEmbedding(100, w2v_model.wv)
    
    for node in test_nodes:
        code_text = node['properties']['code']
        
        # Show what tokenizer produces
        tokens = tokenizer(code_text)
        print(f"\nNode: '{code_text}'")
        print(f"  Tokenized: {tokens}")
        
        # Check if tokens are in vocabulary
        found_tokens = []
        for token in tokens:
            if token in w2v_model.wv:
                found_tokens.append(token)
                print(f"    '{token}': ✓ FOUND in vocabulary")
            else:
                print(f"    '{token}': ❌ NOT FOUND in vocabulary")
        
        # Test embedding
        result = nodes_embedding.embed_nodes([node])
        zero_ratio = (result == 0).float().mean().item()
        
        print(f"  Embedding result:")
        print(f"    Shape: {result.shape}")
        print(f"    Mean: {result.mean().item():.6f}")
        print(f"    Zero ratio: {zero_ratio:.2%}")
        
        if zero_ratio < 0.1:
            print(f"    ✅ SUCCESS: Valid embedding generated!")
        else:
            print(f"    ❌ FAILED: Still producing zeros")
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("If you see '✅ SUCCESS' for most nodes, the fix is working!")
    print("If you still see '❌ FAILED', there may be additional issues.")

if __name__ == "__main__":
    test_tokenization_fix()