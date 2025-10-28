#!/usr/bin/env python3
"""
Debug how to properly access Properties.code()
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')
import src.utils.functions.cpg as cpg
import configs

print("="*80)
print("DEBUGGING PROPERTIES ACCESS")
print("="*80)

# Load a CPG file and parse nodes
cpg_files = list(Path('data/cpg').glob('*_cpg.pkl'))
cpg_df = pd.read_pickle(cpg_files[0])
sample_row = cpg_df.iloc[0]

context = configs.Embed()
nodes = cpg.parse_to_nodes(sample_row['cpg'], context.nodes_dim)

print(f"Testing Properties access on {len(nodes)} nodes...")

code_found = 0
for i, node in enumerate(nodes[:10]):  # Test first 10 nodes
    print(f"\n--- Node {i} ---")
    
    if isinstance(node, dict) and 'properties' in node:
        props = node['properties']
        print(f"Properties type: {type(props)}")
        
        # Try different ways to access code
        try:
            # Method 1: Direct method call
            if hasattr(props, 'code'):
                code_value = props.code()
                print(f"  props.code(): '{code_value}'")
                if code_value:
                    code_found += 1
                    
                    # Test tokenization on this code
                    from src.utils.functions.parse import tokenizer
                    tokens = tokenizer(code_value)
                    print(f"  Tokenized: {tokens}")
                    
                    # Check vocabulary
                    from gensim.models.word2vec import Word2Vec
                    w2v_model = Word2Vec.load('data/w2v/w2v.model')
                    found_tokens = [t for t in tokens if t in w2v_model.wv]
                    print(f"  Found in vocab: {found_tokens}")
                    
                    if found_tokens:
                        print(f"  ✅ This node would produce valid embeddings!")
                        break
            
            # Method 2: Check if it has code
            if hasattr(props, 'has_code'):
                has_code = props.has_code()
                print(f"  props.has_code(): {has_code}")
            
            # Method 3: Direct pairs access
            if hasattr(props, 'pairs'):
                pairs = props.pairs
                print(f"  Available keys: {list(pairs.keys())}")
                if 'CODE' in pairs:
                    code_direct = pairs['CODE']
                    print(f"  Direct CODE: '{code_direct}'")
        
        except Exception as e:
            print(f"  Error accessing properties: {e}")

print(f"\n📊 Summary: Found code in {code_found} out of 10 nodes")

if code_found == 0:
    print("🚨 NO CODE FOUND - This explains the zero embeddings!")
    print("The CPG nodes don't contain actual code text.")
else:
    print("✅ Code found - The issue is in safe_get_node_field access")

print("="*80)