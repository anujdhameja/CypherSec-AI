
"""
Retrain Word2Vec with Fixed Configuration
Run this to regenerate embeddings
"""

import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
import numpy as np

# Configuration
NODES_DIM = 100  # CRITICAL: Must match Word2Vec vector_size

W2V_CONFIG = {
    'vector_size': NODES_DIM,  # MUST MATCH!
    'window': 5,
    'min_count': 3,
    'workers': 4,
    'sg': 1,
    'hs': 0,
    'negative': 10,
    'epochs': 20,
    'alpha': 0.025,
    'min_alpha': 0.0001,
    'seed': 42,
}

def main():
    print("="*80)
    print("RETRAINING WORD2VEC")
    print("="*80)
    
    # Load all token files
    token_files = sorted(Path('data/tokens').glob('*_tokens.pkl'))
    
    print(f"\nFound {len(token_files)} token files")
    
    # Collect all token sequences
    all_token_sequences = []
    
    for i, f in enumerate(token_files):
        df = pd.read_pickle(f)
        sequences = df['tokens'].tolist()
        all_token_sequences.extend(sequences)
        
        if (i+1) % 10 == 0:
            print(f"  Loaded {i+1}/{len(token_files)} files...")
    
    print(f"\nTotal sequences: {len(all_token_sequences)}")
    
    # Train Word2Vec
    print(f"\nTraining Word2Vec with config:")
    for k, v in W2V_CONFIG.items():
        print(f"  {k}: {v}")
    
    model = Word2Vec(sentences=all_token_sequences, **W2V_CONFIG)
    
    print(f"\n✓ Training complete!")
    print(f"  Vocabulary size: {len(model.wv)}")
    print(f"  Vector size: {model.wv.vector_size}")
    
    # Validate
    sample_words = list(model.wv.index_to_key)[:10]
    sample_vectors = np.array([model.wv[w] for w in sample_words])
    
    avg_std = sample_vectors.std(axis=0).mean()
    print(f"  Vector std: {avg_std:.6f}")
    
    if avg_std < 0.01:
        print(f"  ⚠️ WARNING: Vectors still have low variance!")
        print(f"  Try training with more data or different parameters")
    else:
        print(f"  ✓ Vectors look good!")
    
    # Save
    output_path = 'data/w2v/w2v.model'
    model.save(output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    print(f"\n" + "="*80)
    print("NEXT STEP: Re-run embed_task() to regenerate node features")
    print("="*80)

if __name__ == "__main__":
    main()
