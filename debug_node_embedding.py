#!/usr/bin/env python3
"""
Debug Node Embedding Script
Traces the exact point where node features become zeros
"""

import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from gensim.models.word2vec import Word2Vec

# Add src to path
sys.path.append('src')
from src.prepare.embeddings import NodesEmbedding
import src.utils.functions.cpg as cpg
import configs

def debug_node_embedding():
    """Debug the node embedding process step by step"""
    print("=" * 80)
    print("DEBUGGING NODE EMBEDDING PROCESS")
    print("=" * 80)
    
    test_output_dir = Path("data/test_output")
    
    # Step 1: Load test data
    print("\n🔄 STEP 1: Loading test data...")
    
    try:
        # Load tokens
        tokens_df = pd.read_pickle(test_output_dir / "test_tokens.pkl")
        print(f"✓ Loaded tokens: {len(tokens_df)} samples")
        
        # Load Word2Vec model
        w2v_model = Word2Vec.load(str(test_output_dir / "test_w2v.model"))
        print(f"✓ Loaded Word2Vec model with {len(w2v_model.wv.key_to_index)} words")
        
        # Load CPG data
        cpg_df = pd.read_pickle(test_output_dir / "test_cpg.pkl")
        print(f"✓ Loaded CPG data: {len(cpg_df)} samples")
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Step 2: Focus on first sample
    print("\n🔄 STEP 2: Analyzing first sample...")
    
    sample_idx = 0
    sample_tokens = tokens_df['tokens'].iloc[sample_idx]
    sample_cpg = cpg_df.iloc[sample_idx]
    
    print(f"Sample {sample_idx}:")
    print(f"  Target: {sample_cpg['target']}")
    print(f"  Function: {sample_cpg['func'][:100]}...")
    print(f"  Tokens ({len(sample_tokens)}): {sample_tokens[:10]}...")
    
    # Step 3: Test Word2Vec embeddings manually
    print("\n🔄 STEP 3: Testing Word2Vec embeddings manually...")
    
    print("Manual Word2Vec lookups:")
    for i, token in enumerate(sample_tokens[:5]):
        if token in w2v_model.wv:
            embedding = w2v_model.wv[token]
            print(f"  Token '{token}': shape={embedding.shape}, mean={embedding.mean():.6f}, "
                  f"std={embedding.std():.6f}, first_5={embedding[:5]}")
        else:
            print(f"  Token '{token}': NOT FOUND in vocabulary")
    
    # Step 4: Parse CPG to nodes
    print("\n🔄 STEP 4: Parsing CPG to nodes...")
    
    try:
        context = configs.Embed()
        sample_cpg_dict = sample_cpg['cpg']
        
        print(f"CPG structure: {type(sample_cpg_dict)}")
        print(f"CPG keys: {list(sample_cpg_dict.keys()) if isinstance(sample_cpg_dict, dict) else 'Not a dict'}")
        
        # Parse to nodes
        nodes = cpg.parse_to_nodes(sample_cpg_dict, context.nodes_dim)
        print(f"✓ Parsed {len(nodes)} nodes from CPG")
        
        # Show first node structure
        if nodes:
            first_node = nodes[0]
            node_dict = first_node if isinstance(first_node, dict) else first_node.__dict__
            print(f"First node keys: {list(node_dict.keys())}")
            print(f"First node id: {node_dict.get('id')}")
            print(f"First node label: {node_dict.get('label')}")
            print(f"First node type: {node_dict.get('type')}")
            
            # Check properties
            properties = node_dict.get('properties', {})
            print(f"First node properties: {properties}")
            
    except Exception as e:
        print(f"❌ Error parsing CPG to nodes: {e}")
        return
    
    # Step 5: Test NodesEmbedding class
    print("\n🔄 STEP 5: Testing NodesEmbedding class...")
    
    try:
        # Create NodesEmbedding instance
        nodes_embedding = NodesEmbedding(w2v_model.wv, context.nodes_dim)
        print(f"✓ Created NodesEmbedding with dim={context.nodes_dim}")
        
        # Test embedding on first few nodes
        print("\nTesting individual node embeddings:")
        
        for i, node in enumerate(nodes[:3]):
            print(f"\n--- Node {i} ---")
            node_dict = node if isinstance(node, dict) else node.__dict__
            
            # Show what text will be extracted
            print(f"Node id: {node_dict.get('id')}")
            print(f"Node label: {node_dict.get('label')}")
            print(f"Node type: {node_dict.get('type')}")
            
            # Check properties for text content
            properties = node_dict.get('properties', {})
            print(f"Properties keys: {list(properties.keys()) if properties else 'None'}")
            
            # Look for text fields
            text_fields = ['code', 'name', 'fullName', 'signature', 'value']
            found_text = []
            for field in text_fields:
                if field in properties:
                    value = properties[field]
                    if value and str(value).strip():
                        found_text.append(f"{field}='{value}'")
            
            print(f"Text content found: {found_text}")
            
            # Try to embed this node
            try:
                node_embedding = nodes_embedding.embed(node)
                print(f"Node embedding result:")
                print(f"  Shape: {node_embedding.shape}")
                print(f"  Mean: {node_embedding.mean():.6f}")
                print(f"  Std: {node_embedding.std():.6f}")
                print(f"  Zero ratio: {(node_embedding == 0).mean():.2%}")
                print(f"  First 5 values: {node_embedding[:5]}")
                
                if (node_embedding == 0).all():
                    print(f"  ⚠️  ALL ZEROS! This node embedding failed!")
                else:
                    print(f"  ✓ Node embedding has non-zero values")
                    
            except Exception as e:
                print(f"  ❌ Error embedding node: {e}")
        
    except Exception as e:
        print(f"❌ Error with NodesEmbedding: {e}")
        return
    
    # Step 6: Deep dive into NodesEmbedding.embed() method
    print("\n🔄 STEP 6: Deep dive into NodesEmbedding.embed() method...")
    
    try:
        # Let's manually trace through the embed method
        first_node = nodes[0]
        node_dict = first_node if isinstance(first_node, dict) else first_node.__dict__
        
        print(f"Manually tracing embed() for first node...")
        print(f"Node structure: {type(first_node)}")
        
        # Check what safe_get_node_field returns
        from src.prepare.embeddings import safe_get_node_field
        
        # Test different field extraction methods
        test_fields = ['code', 'name', 'fullName', 'signature', 'value', 'label', 'type']
        
        print(f"\nTesting safe_get_node_field() on different fields:")
        for field in test_fields:
            try:
                result = safe_get_node_field(first_node, field)
                print(f"  {field}: '{result}' (type: {type(result)})")
            except Exception as e:
                print(f"  {field}: ERROR - {e}")
        
        # Check the actual embedding process
        print(f"\nManual embedding process:")
        
        # Get the text that would be used for embedding
        node_text = safe_get_node_field(first_node, 'code') or \
                   safe_get_node_field(first_node, 'name') or \
                   safe_get_node_field(first_node, 'fullName') or \
                   safe_get_node_field(first_node, 'signature') or \
                   safe_get_node_field(first_node, 'label') or \
                   safe_get_node_field(first_node, 'type') or ""
        
        print(f"Extracted text for embedding: '{node_text}'")
        
        if node_text and node_text.strip():
            # Check if this text is in Word2Vec vocabulary
            if node_text in w2v_model.wv:
                manual_embedding = w2v_model.wv[node_text]
                print(f"Manual W2V lookup successful:")
                print(f"  Shape: {manual_embedding.shape}")
                print(f"  Mean: {manual_embedding.mean():.6f}")
                print(f"  First 5: {manual_embedding[:5]}")
            else:
                print(f"Text '{node_text}' NOT FOUND in Word2Vec vocabulary!")
                print(f"Available vocabulary (first 10): {list(w2v_model.wv.key_to_index.keys())[:10]}")
        else:
            print(f"No text extracted from node - this will result in zero embedding!")
        
    except Exception as e:
        print(f"❌ Error in deep dive: {e}")
    
    # Step 7: Compare with full pipeline
    print("\n🔄 STEP 7: Comparing with full pipeline result...")
    
    try:
        # Use the same function as the main pipeline
        import src.prepare as prepare
        
        print(f"Running full nodes_to_input() function...")
        
        result = prepare.nodes_to_input(
            nodes, 
            sample_cpg['target'], 
            context.nodes_dim, 
            w2v_model.wv, 
            context.edge_type
        )
        
        print(f"Full pipeline result:")
        print(f"  Type: {type(result)}")
        if hasattr(result, 'x'):
            print(f"  Features shape: {result.x.shape}")
            print(f"  Features mean: {result.x.mean().item():.6f}")
            print(f"  Features std: {result.x.std().item():.6f}")
            print(f"  Zero ratio: {(result.x == 0).float().mean().item():.2%}")
            print(f"  First node features: {result.x[0][:5]}")
            
            if (result.x == 0).all():
                print(f"  🚨 CONFIRMED: Full pipeline produces ALL ZEROS!")
            else:
                print(f"  ✓ Full pipeline produces non-zero features")
        
    except Exception as e:
        print(f"❌ Error running full pipeline: {e}")
    
    # Step 8: Final diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS AND CONCLUSIONS")
    print("=" * 80)
    
    print(f"\n🔍 ROOT CAUSE ANALYSIS:")
    print(f"1. Word2Vec model: {'✓ Working' if len(w2v_model.wv.key_to_index) > 0 else '❌ Broken'}")
    print(f"2. CPG parsing: {'✓ Working' if len(nodes) > 0 else '❌ Broken'}")
    print(f"3. Node structure: {'✓ Has content' if nodes and len(nodes) > 0 else '❌ Empty'}")
    
    # Check vocabulary mismatch
    if nodes and len(nodes) > 0:
        first_node = nodes[0]
        node_dict = first_node if isinstance(first_node, dict) else first_node.__dict__
        properties = node_dict.get('properties', {})
        
        print(f"\n🔍 VOCABULARY MISMATCH CHECK:")
        print(f"Available W2V vocabulary: {list(w2v_model.wv.key_to_index.keys())}")
        print(f"Node properties: {properties}")
        
        # Check if any node property values match vocabulary
        vocab_matches = []
        for key, value in properties.items():
            if str(value) in w2v_model.wv.key_to_index:
                vocab_matches.append(f"{key}='{value}'")
        
        if vocab_matches:
            print(f"✓ Found vocabulary matches: {vocab_matches}")
        else:
            print(f"❌ NO vocabulary matches found!")
            print(f"This explains why all embeddings are zero!")
    
    print(f"\n💡 RECOMMENDED FIXES:")
    print(f"1. Check safe_get_node_field() - is it extracting the right text?")
    print(f"2. Check vocabulary alignment - do node texts match W2V vocabulary?")
    print(f"3. Check NodesEmbedding.embed() - is it handling missing vocabulary correctly?")
    print(f"4. Add fallback embeddings for unknown tokens")

if __name__ == "__main__":
    debug_node_embedding()