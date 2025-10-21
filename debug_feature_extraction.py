"""
Diagnose where feature quality degrades in the preprocessing pipeline.
"""
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from src.prepare.embeddings import NodesEmbedding

print("="*80)
print("FEATURE EXTRACTION PIPELINE DEBUG")
print("="*80)

# Load Word2Vec model
w2v_path = Path('data/w2v/w2v.model')
if w2v_path.exists():
    w2v = Word2Vec.load(str(w2v_path))
    print(f"\n✓ Word2Vec model loaded")
    print(f"  Vocabulary: {len(w2v.wv)}")
    print(f"  Vector size: {w2v.vector_size}")
else:
    print(f"\n✗ Word2Vec model not found at {w2v_path}")
    exit(1)

# Load token file to see what text should be extracted
print("\n" + "="*80)
print("CHECKING TOKEN DATA (what should be embedded)")
print("="*80)

token_path = Path('data/tokens/0_cpg_tokens.pkl')
if token_path.exists():
    token_df = pd.read_pickle(token_path)
    print(f"\nToken file shape: {token_df.shape}")
    print(f"Columns: {token_df.columns.tolist()}")
    
    # Show sample tokens
    print(f"\nSample tokens from first row:")
    sample_tokens = token_df.iloc[0]['tokens']
    print(f"  Type: {type(sample_tokens)}")
    print(f"  Length: {len(sample_tokens)}")
    print(f"  First 20: {sample_tokens[:20]}")
    
    # Check which tokens are in W2V vocab
    in_vocab = sum(1 for t in sample_tokens if t in w2v.wv)
    print(f"  In W2V vocab: {in_vocab}/{len(sample_tokens)} ({in_vocab/len(sample_tokens)*100:.1f}%)")

# Load CPG JSON to see node text
print("\n" + "="*80)
print("CHECKING CPG DATA (raw node content)")
print("="*80)

cpg_path = Path('data/cpg/0_cpg.pkl')
if cpg_path.exists():
    import pandas as pd
    cpg_df = pd.read_pickle(cpg_path)
    print(f"\nCPG DataFrame shape: {cpg_df.shape}")
    print(f"CPG DataFrame columns: {list(cpg_df.columns)}")
    
    if len(cpg_df) > 0:
        sample_cpg = cpg_df.iloc[0]['cpg']
        print(f"\nSample CPG type: {type(sample_cpg)}")
        
        if isinstance(sample_cpg, dict) and 'functions' in sample_cpg:
            first_function = sample_cpg['functions'][0]
            nodes = first_function.get('nodes', [])
            print(f"\nFirst function has {len(nodes)} nodes")
            print(f"First 10 nodes from CPG:")
            for i, node in enumerate(nodes[:10]):
                code = node.get('code', 'N/A')
                label = node.get('label', 'N/A')
                code_preview = (code[:50] + '...') if len(str(code)) > 50 else code
                print(f"  {i}: label={label:12} code='{code_preview}'")

# Now check the actual PyG data
print("\n" + "="*80)
print("CHECKING FINAL PyG DATA (what model receives)")
print("="*80)

input_path = Path('data/input/0_cpg_input.pkl')
if input_path.exists():
    df = pd.read_pickle(input_path)
    graph = df.iloc[0]['input']
    
    print(f"\nFirst graph from PyG data:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Features shape: {graph.x.shape}")
    print(f"  Feature mean: {graph.x.mean(dim=0)[:10]}")
    print(f"  Feature std: {graph.x.std(dim=0)[:10]}")
    
    # Check if features vary
    print(f"\nFeature variation analysis:")
    unique_rows = len(torch.unique(graph.x, dim=0))
    print(f"  Unique feature vectors: {unique_rows}/{graph.num_nodes}")
    
    # Check feature statistics per node
    node_magnitudes = torch.norm(graph.x, dim=1)
    print(f"  Node magnitude (L2 norm): mean={node_magnitudes.mean():.4f}, std={node_magnitudes.std():.4f}")
    print(f"  Range: [{node_magnitudes.min():.4f}, {node_magnitudes.max():.4f}]")
    
    # Compare two random nodes
    if graph.num_nodes >= 2:
        diff = torch.norm(graph.x[0] - graph.x[1])
        print(f"  Distance between node 0 and 1: {diff:.4f}")

# Now let's manually test the embedding process
print("\n" + "="*80)
print("MANUAL EMBEDDING TEST")
print("="*80)

# Create a simple test case
test_tokens = [['if', 'else', 'for', 'while'], ['int', 'char', 'void', 'return']]
print(f"\nTest tokens: {test_tokens}")

try:
    # Fix: NodesEmbedding expects (nodes_dim, keyed_vectors)
    embedder = NodesEmbedding(nodes_dim=100, keyed_vectors=w2v.wv)
    
    for i, tokens in enumerate(test_tokens):
        # Create fake nodes with these tokens as code
        fake_nodes = {f"node_{j}": {"code": token} for j, token in enumerate(tokens)}
        
        embeddings = embedder(fake_nodes)
        print(f"\nTest {i+1}: {tokens}")
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Mean: {embeddings.mean(dim=0)[:5]}")
        print(f"  Std: {embeddings.std(dim=0)[:5]}")
        
        # Check if embeddings are different
        if len(embeddings) > 1:
            dist = torch.norm(embeddings[0] - embeddings[1])
            print(f"  Distance between first two: {dist:.4f}")

except Exception as e:
    print(f"Error during embedding: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
Key questions answered:
1. Are tokens in W2V vocab? (should be 90%+)
2. Do CPG nodes have meaningful text? (should be code keywords)
3. Are final features actually W2Vec embeddings? (should vary by node)
4. Are embeddings properly normalized? (should have consistent magnitude)

If any of these fail, that's where quality degrades.
""")