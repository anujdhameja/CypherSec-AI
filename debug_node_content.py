#!/usr/bin/env python3
"""
Debug what's actually in the nodes and why embeddings are zero
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')
import src.utils.functions.cpg as cpg
import configs

print("="*80)
print("DEBUGGING NODE CONTENT")
print("="*80)

# Load a CPG file
cpg_files = list(Path('data/cpg').glob('*_cpg.pkl'))
if not cpg_files:
    print("❌ No CPG files found")
    exit(1)

print(f"📁 Loading CPG file: {cpg_files[0].name}")
cpg_df = pd.read_pickle(cpg_files[0])

# Get first sample
sample_row = cpg_df.iloc[0]
print(f"\n🔍 Sample structure:")
print(f"  Columns: {list(sample_row.index)}")
print(f"  Target: {sample_row['target']}")
print(f"  Function preview: {str(sample_row['func'])[:100]}...")

# Check CPG structure
cpg_data = sample_row['cpg']
print(f"\n🔍 CPG structure:")
print(f"  Type: {type(cpg_data)}")
if isinstance(cpg_data, dict):
    print(f"  Keys: {list(cpg_data.keys())}")
    
    # Parse to nodes
    context = configs.Embed()
    try:
        nodes = cpg.parse_to_nodes(cpg_data, context.nodes_dim)
        print(f"  Parsed {len(nodes)} nodes")
        
        # Check first few nodes
        for i, node in enumerate(nodes[:3]):
            print(f"\n--- Node {i} ---")
            
            if isinstance(node, dict):
                print(f"  Keys: {list(node.keys())}")
                
                # Check properties
                if 'properties' in node:
                    props = node['properties']
                    print(f"  Properties type: {type(props)}")
                    
                    if isinstance(props, dict):
                        print(f"  Properties keys: {list(props.keys())}")
                        
                        # Show actual content
                        for key in ['code', 'label', 'name', 'type', 'value']:
                            if key in props:
                                value = props[key]
                                print(f"    {key}: '{value}' (type: {type(value)})")
                    else:
                        print(f"  Properties: {props}")
                
                # Check other fields
                for key in ['label', 'type', 'id']:
                    if key in node:
                        print(f"  {key}: '{node[key]}'")
            else:
                print(f"  Node type: {type(node)}")
                if hasattr(node, '__dict__'):
                    print(f"  Attributes: {list(node.__dict__.keys())}")
        
        # Test safe_get_node_field on first node
        if nodes:
            from src.prepare.embeddings import safe_get_node_field
            from src.utils.functions.parse import tokenizer
            
            first_node = nodes[0]
            print(f"\n🔍 Testing safe_get_node_field on first node:")
            
            extracted_text = None
            for field in ['code', 'label', 'name', 'type', 'value']:
                try:
                    value = safe_get_node_field(first_node, field)
                    print(f"  {field}: '{value}' (type: {type(value)})")
                    if value and isinstance(value, str) and len(value.strip()) > 0:
                        if extracted_text is None:
                            extracted_text = value
                except Exception as e:
                    print(f"  {field}: ERROR - {e}")
            
            print(f"\n🔍 Final extracted text: '{extracted_text}'")
            
            if extracted_text:
                tokens = tokenizer(extracted_text)
                print(f"  Tokenized: {tokens}")
                
                # Load Word2Vec and check
                from gensim.models.word2vec import Word2Vec
                try:
                    w2v_model = Word2Vec.load('data/w2v/w2v.model')
                    found_count = 0
                    for token in tokens:
                        if token in w2v_model.wv:
                            found_count += 1
                            print(f"    '{token}': ✓ FOUND")
                        else:
                            print(f"    '{token}': ❌ NOT FOUND")
                    
                    print(f"  Found {found_count}/{len(tokens)} tokens in vocabulary")
                    
                    if found_count == 0:
                        print(f"  🚨 NO TOKENS FOUND - This explains zero embeddings!")
                        print(f"  Vocabulary sample: {list(w2v_model.wv.index_to_key)[:20]}")
                    
                except Exception as e:
                    print(f"  Error loading Word2Vec: {e}")
            else:
                print(f"  🚨 NO TEXT EXTRACTED - This explains zero embeddings!")
        
    except Exception as e:
        print(f"❌ Error parsing CPG to nodes: {e}")
        import traceback
        traceback.print_exc()

print("="*80)