"""
FIXED Configuration for Word2Vec
This will create meaningful embeddings
"""

class Embed:
    """
    Fixed embedding configuration
    
    CRITICAL CHANGES:
    1. Match nodes_dim to W2V vector_size
    2. Increase min_count (vocabulary filtering)
    3. More training iterations
    4. Better window size
    """
    
    nodes_dim = 100  # MUST MATCH vector_size!
    
    edge_type = 'Ast'
    
    # Word2Vec arguments - FIXED
    w2v_args = {
        'vector_size': 100,      # MATCHES nodes_dim!
        'window': 5,             # Context window
        'min_count': 3,          # Ignore rare words (was probably 1)
        'workers': 4,            # Parallel processing
        'sg': 1,                 # Skip-gram (better for code)
        'hs': 0,                 # Negative sampling
        'negative': 10,          # Number of negative samples
        'epochs': 20,            # More epochs (was 5)
        'alpha': 0.025,          # Initial learning rate
        'min_alpha': 0.0001,     # Final learning rate
        'seed': 42,              # Reproducibility
    }
    
    # Add validation
    def __post_init__(self):
        assert self.nodes_dim == self.w2v_args['vector_size'], \
            f"nodes_dim ({self.nodes_dim}) must match vector_size ({self.w2v_args['vector_size']})"
        print(f"✓ Embed config validated: nodes_dim={self.nodes_dim}")


class EmbedAlternative:
    """
    Alternative: Use 200-dim embeddings (closer to original 205)
    """
    
    nodes_dim = 200  # Higher dimensional
    
    edge_type = 'Ast'
    
    w2v_args = {
        'vector_size': 200,      # MATCHES nodes_dim!
        'window': 7,             # Larger context
        'min_count': 2,          # Keep more words
        'workers': 4,
        'sg': 1,
        'hs': 0,
        'negative': 15,
        'epochs': 30,            # More training
        'alpha': 0.025,
        'min_alpha': 0.0001,
        'seed': 42,
    }


# ============================================
# Diagnostic: Check if W2V will work
# ============================================

def validate_word2vec_training(tokens_path='data/tokens'):
    """
    Check if you have enough data for Word2Vec to work
    """
    import pandas as pd
    from pathlib import Path
    from collections import Counter
    
    print("\n" + "="*80)
    print("WORD2VEC TRAINING VALIDATION")
    print("="*80)
    
    # Load token files
    token_files = list(Path(tokens_path).glob('*_tokens.pkl'))
    
    if not token_files:
        print("❌ No token files found!")
        return False
    
    print(f"\n1. Data Check:")
    print(f"   Token files: {len(token_files)}")
    
    # Collect all tokens
    all_tokens = []
    total_functions = 0
    
    for f in token_files[:5]:  # Check first 5 files
        df = pd.read_pickle(f)
        for tokens in df['tokens']:
            all_tokens.extend(tokens)
        total_functions += len(df)
    
    print(f"   Functions sampled: {total_functions}")
    print(f"   Total tokens: {len(all_tokens)}")
    
    # Vocabulary analysis
    vocab = Counter(all_tokens)
    
    print(f"\n2. Vocabulary Analysis:")
    print(f"   Unique tokens: {len(vocab)}")
    print(f"   Most common 10:")
    for token, count in vocab.most_common(10):
        print(f"      '{token}': {count}")
    
    # Check if enough data
    min_tokens = 50000  # Rule of thumb
    min_vocab = 500
    
    issues = []
    
    if len(all_tokens) < min_tokens:
        issues.append(f"Not enough tokens ({len(all_tokens)} < {min_tokens})")
    else:
        print(f"   ✓ Sufficient tokens: {len(all_tokens)}")
    
    if len(vocab) < min_vocab:
        issues.append(f"Vocabulary too small ({len(vocab)} < {min_vocab})")
    else:
        print(f"   ✓ Good vocabulary size: {len(vocab)}")
    
    # Check token distribution
    singleton_count = sum(1 for count in vocab.values() if count == 1)
    singleton_ratio = singleton_count / len(vocab)
    
    print(f"\n3. Token Distribution:")
    print(f"   Singletons: {singleton_count} ({singleton_ratio:.1%})")
    
    if singleton_ratio > 0.5:
        print(f"   ⚠️ Many rare tokens - consider increasing min_count")
    
    if issues:
        print(f"\n❌ Issues Found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✓ Data looks good for Word2Vec training!")
        return True


# ============================================
# Script to retrain Word2Vec
# ============================================

def retrain_word2vec():
    """
    Complete script to retrain Word2Vec with correct settings
    """
    script = '''
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
    
    print(f"\\nFound {len(token_files)} token files")
    
    # Collect all token sequences
    all_token_sequences = []
    
    for i, f in enumerate(token_files):
        df = pd.read_pickle(f)
        sequences = df['tokens'].tolist()
        all_token_sequences.extend(sequences)
        
        if (i+1) % 10 == 0:
            print(f"  Loaded {i+1}/{len(token_files)} files...")
    
    print(f"\\nTotal sequences: {len(all_token_sequences)}")
    
    # Train Word2Vec
    print(f"\\nTraining Word2Vec with config:")
    for k, v in W2V_CONFIG.items():
        print(f"  {k}: {v}")
    
    model = Word2Vec(sentences=all_token_sequences, **W2V_CONFIG)
    
    print(f"\\n✓ Training complete!")
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
    print(f"\\n✓ Saved to: {output_path}")
    
    print(f"\\n" + "="*80)
    print("NEXT STEP: Re-run embed_task() to regenerate node features")
    print("="*80)

if __name__ == "__main__":
    main()
'''
    
    with open('retrain_word2vec.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("\n✓ Created: retrain_word2vec.py")
    print("Run: python retrain_word2vec.py")


if __name__ == "__main__":
    print(__doc__)
    
    # Validate data
    if validate_word2vec_training():
        print("\n✓ Ready to retrain Word2Vec")
        retrain_word2vec()
    else:
        print("\n⚠️ Data issues found - fix them before retraining")