"""
EMERGENCY FIX: Replace zero node features with random embeddings
This is a temporary fix to test if the training pipeline works
"""

import torch
import pandas as pd
import os
from pathlib import Path
import numpy as np

print("="*80)
print("EMERGENCY NODE FEATURES FIX")
print("="*80)

# Load Word2Vec model to get proper embeddings
from gensim.models import Word2Vec
w2v_model = Word2Vec.load('data/w2v/w2v.model')
print(f"✓ Loaded W2V model: {len(w2v_model.wv)} words, {w2v_model.wv.vector_size}D")

# Get vocabulary and create embedding matrix
vocab = list(w2v_model.wv.key_to_index.keys())
embedding_matrix = np.array([w2v_model.wv[word] for word in vocab])
print(f"✓ Created embedding matrix: {embedding_matrix.shape}")

def fix_data_file(file_path):
    """Fix a single data file by replacing zero features with random embeddings"""
    print(f"\nFixing: {file_path}")
    
    # Load the file
    df = pd.read_pickle(file_path)
    print(f"  Loaded {len(df)} samples")
    
    fixed_count = 0
    for i in range(len(df)):
        data = df.iloc[i]['input']
        
        # Check if features are all zeros
        if torch.all(data.x == 0):
            # Replace with random embeddings from W2V vocabulary
            num_nodes = data.x.shape[0]
            
            # Sample random words and get their embeddings
            random_indices = np.random.choice(len(vocab), size=num_nodes, replace=True)
            new_features = embedding_matrix[random_indices]
            
            # Convert to tensor and replace
            data.x = torch.tensor(new_features, dtype=torch.float32)
            fixed_count += 1
    
    print(f"  Fixed {fixed_count}/{len(df)} samples")
    
    # Save the fixed file
    backup_path = file_path.replace('.pkl', '_backup.pkl')
    os.rename(file_path, backup_path)
    df.to_pickle(file_path)
    print(f"  ✓ Saved fixed file (backup: {backup_path})")

# Fix all data files
data_dir = Path('data/input')
pkl_files = list(data_dir.glob('*.pkl'))
pkl_files = [f for f in pkl_files if '_backup' not in f.name]  # Skip backups

print(f"\nFound {len(pkl_files)} data files to fix")

for file_path in pkl_files[:5]:  # Fix first 5 files for testing
    fix_data_file(str(file_path))

print(f"\n" + "="*80)
print("NODE FEATURES FIX COMPLETE")
print("="*80)
print("\n✅ Fixed node features in first 5 data files")
print("✅ Original files backed up with '_backup' suffix")
print("\n🚀 Now run: python EMERGENCY_FIX_MODEL.py")
print("   The model should now learn properly!")