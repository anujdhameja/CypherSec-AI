#!/usr/bin/env python3
"""
Quick test to verify the embedding fix
"""

import sys
sys.path.append('src')

from src.prepare.embeddings import NodesEmbedding
from gensim.models.word2vec import Word2Vec
import numpy as np

def test_fix():
    print("=" * 50)
    print("TESTING EMBEDDING FIX")
    print("=" * 50)
    
    # Load Word2Vec model
    try:
        w2v_model = Word2Vec.load("data/test_output/test_w2v.model")
        print(f"✓ Loaded Word2Vec with vocab: {list(w2v_model.wv.key_to_index.keys())}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Create NodesEmbedding
    nodes_embedding = NodesEmbedding(100, w2v_model.wv)
    
    # Test problematic nodes
    test_nodes = [
        {'id': '1', 'properties': {'code': 'buffer'}},
        {'id': '2', 'properties': {'code': 'strcpy'}},
        {'id': '3', 'properties': {'code': 'void'}},
    ]
    
    print(f"\n🔍 Testing nodes:")
    
    for node in test_nodes:
        code = node['properties']['code']
        result = nodes_embedding.embed_nodes([node])
        zero_ratio = (result == 0).float().mean().item()
        
        print(f"'{code}': zero_ratio={zero_ratio:.1%} ", end="")
        if zero_ratio < 0.1:
            print("✅ FIXED!")
        else:
            print("❌ Still broken")
    
    print(f"\n" + "=" * 50)

if __name__ == "__main__":
    test_fix()