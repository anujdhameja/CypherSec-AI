"""
Diagnose Word2Vec Embeddings
Check if your node features are meaningful
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from collections import Counter


def check_node_features():
    """Check if node features have any variance"""
    print("\n" + "="*80)
    print("NODE FEATURE QUALITY CHECK")
    print("="*80)
    
    # Load some input data
    input_files = list(Path('data/input').glob('*_input.pkl'))
    
    if not input_files:
        print("❌ No input files found")
        return
    
    # Load first file
    df = pd.read_pickle(input_files[0])
    
    print(f"\n1. Basic Stats:")
    print(f"   Samples in file: {len(df)}")
    
    # Collect node features from multiple graphs
    all_node_features = []
    graph_node_counts = []
    
    for i in range(min(100, len(df))):
        graph = df.iloc[i]['input']
        features = graph.x.numpy()  # [num_nodes, 205]
        
        all_node_features.append(features)
        graph_node_counts.append(len(features))
    
    # Concatenate all node features
    all_features = np.vstack(all_node_features)
    
    print(f"\n2. Node Feature Statistics:")
    print(f"   Graphs sampled: {len(all_node_features)}")
    print(f"   Total nodes: {len(all_features)}")
    print(f"   Feature dimension: {all_features.shape[1]}")
    print(f"   Avg nodes per graph: {np.mean(graph_node_counts):.1f}")
    
    # Check feature statistics
    feature_means = all_features.mean(axis=0)
    feature_stds = all_features.std(axis=0)
    
    print(f"\n3. Feature Distribution:")
    print(f"   Global mean: {all_features.mean():.6f}")
    print(f"   Global std: {all_features.std():.6f}")
    print(f"   Min value: {all_features.min():.6f}")
    print(f"   Max value: {all_features.max():.6f}")
    
    # Check per-dimension variance
    zero_variance_dims = (feature_stds < 1e-6).sum()
    low_variance_dims = (feature_stds < 0.01).sum()
    
    print(f"\n4. Dimension-wise Variance:")
    print(f"   Zero variance dimensions: {zero_variance_dims}/{len(feature_stds)}")
    print(f"   Low variance (<0.01) dimensions: {low_variance_dims}/{len(feature_stds)}")
    
    # Check if features are all zeros or all same
    if all_features.std() < 1e-6:
        print(f"\n❌ CRITICAL: All features are essentially ZERO or CONSTANT!")
        print(f"   This explains why model can't learn.")
        return False
    
    if zero_variance_dims > len(feature_stds) * 0.8:
        print(f"\n⚠️ WARNING: {zero_variance_dims} dimensions have zero variance!")
        print(f"   Most Word2Vec dimensions are not being used.")
    
    # Check if different nodes have different features
    print(f"\n5. Node Diversity Check:")
    
    # Sample some nodes and check uniqueness
    sample_nodes = all_features[:100]
    unique_nodes = np.unique(sample_nodes, axis=0)
    
    print(f"   Sampled 100 nodes")
    print(f"   Unique feature vectors: {len(unique_nodes)}")
    
    if len(unique_nodes) < 10:
        print(f"   ❌ CRITICAL: Most nodes have identical features!")
        return False
    elif len(unique_nodes) < 50:
        print(f"   ⚠️ WARNING: Low node diversity")
    else:
        print(f"   ✓ Good node diversity")
    
    # Check specific graphs
    print(f"\n6. Per-Graph Analysis:")
    
    for i in range(min(5, len(df))):
        graph = df.iloc[i]['input']
        features = graph.x.numpy()
        target = df.iloc[i]['target']
        
        graph_std = features.std()
        unique_nodes_in_graph = len(np.unique(features, axis=0))
        
        print(f"   Graph {i} (target={target}):")
        print(f"      Nodes: {len(features)}")
        print(f"      Feature std: {graph_std:.6f}")
        print(f"      Unique node types: {unique_nodes_in_graph}")
        
        if graph_std < 1e-6:
            print(f"      ❌ All nodes in this graph have IDENTICAL features!")
    
    return True


def check_word2vec_model():
    """Check the Word2Vec model itself"""
    print("\n" + "="*80)
    print("WORD2VEC MODEL CHECK")
    print("="*80)
    
    try:
        from gensim.models import Word2Vec
        
        w2v_path = Path('data/w2v/w2v.model')
        if not w2v_path.exists():
            print("❌ Word2Vec model not found at data/w2v/w2v.model")
            return
        
        model = Word2Vec.load(str(w2v_path))
        
        print(f"\n1. Model Parameters:")
        print(f"   Vector size: {model.wv.vector_size}")
        print(f"   Vocabulary size: {len(model.wv)}")
        print(f"   Training epochs: {model.epochs}")
        
        # Check vocabulary
        vocab_sample = list(model.wv.index_to_key)[:20]
        print(f"\n2. Sample Vocabulary (first 20 words):")
        for word in vocab_sample:
            print(f"      '{word}'")
        
        # Check if vectors are meaningful
        if len(model.wv) > 10:
            sample_words = list(model.wv.index_to_key)[:10]
            sample_vectors = [model.wv[word] for word in sample_words]
            
            vectors_array = np.array(sample_vectors)
            avg_std = vectors_array.std(axis=0).mean()
            
            print(f"\n3. Vector Quality:")
            print(f"   Average std across dimensions: {avg_std:.6f}")
            
            if avg_std < 0.01:
                print(f"   ❌ CRITICAL: Word vectors have very low variance!")
                print(f"   Word2Vec may not have trained properly")
            else:
                print(f"   ✓ Vectors have reasonable variance")
        
        # Check for common tokens
        common_tokens = ['if', 'for', 'while', 'return', 'int', 'void']
        found_tokens = [t for t in common_tokens if t in model.wv]
        
        print(f"\n4. Common Code Tokens:")
        print(f"   Looking for: {common_tokens}")
        print(f"   Found: {found_tokens}")
        
        if len(found_tokens) == 0:
            print(f"   ⚠️ WARNING: No common code tokens found")
            print(f"   Tokenization might be wrong")
        
    except Exception as e:
        print(f"❌ Error loading Word2Vec model: {e}")


def check_tokens():
    """Check the tokenized data"""
    print("\n" + "="*80)
    print("TOKENIZATION CHECK")
    print("="*80)
    
    token_files = list(Path('data/tokens').glob('*_cpg_tokens.pkl'))
    
    if not token_files:
        print("❌ No token files found")
        return
    
    # Load first token file
    df = pd.read_pickle(token_files[0])
    
    print(f"\n1. Token File Stats:")
    print(f"   Samples: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Check tokens
    if 'tokens' in df.columns:
        sample_tokens = df['tokens'].iloc[:5]
        
        print(f"\n2. Sample Tokenizations:")
        for i, tokens in enumerate(sample_tokens):
            print(f"   Sample {i}: {len(tokens)} tokens")
            print(f"      First 10: {tokens[:10]}")
        
        # Check token counts
        all_token_counts = [len(t) for t in df['tokens']]
        
        print(f"\n3. Token Statistics:")
        print(f"   Avg tokens per function: {np.mean(all_token_counts):.1f}")
        print(f"   Min tokens: {min(all_token_counts)}")
        print(f"   Max tokens: {max(all_token_counts)}")
        
        # Check for empty token lists
        empty_count = sum(1 for t in df['tokens'] if len(t) == 0)
        if empty_count > 0:
            print(f"   ⚠️ WARNING: {empty_count} samples have zero tokens!")


def main():
    print("\n" + "#"*80)
    print("# EMBEDDING QUALITY DIAGNOSTIC")
    print("#"*80)
    
    # Check tokens
    check_tokens()
    
    # Check Word2Vec model
    check_word2vec_model()
    
    # Check node features
    features_ok = check_node_features()
    
    # Final diagnosis
    print("\n" + "="*80)
    print("FINAL DIAGNOSIS")
    print("="*80)
    
    if not features_ok:
        print("\n❌ CRITICAL ISSUE FOUND:")
        print("   Your node features have no variance or are all identical.")
        print("\n💡 This is why no model can learn from your data!")
        print("\n🔧 Possible causes:")
        print("   1. Word2Vec model didn't train properly")
        print("      → Check if you have enough training data")
        print("      → Check if tokenization is working")
        print("   2. NodesEmbedding is not extracting features correctly")
        print("      → Check safe_get_node_field() function")
        print("      → Check if 'code' or 'label' fields exist in nodes")
        print("   3. All code snippets are too similar")
        print("      → Unlikely, but check your source data")
    else:
        print("\n✓ Node features look reasonable")
        print("   The problem might be elsewhere:")
        print("   - Graph structure (edges)")
        print("   - Model architecture")
        print("   - Training procedure")


if __name__ == "__main__":
    main()