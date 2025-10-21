import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path

print("="*80)
print("BATCH MIXING CHECK")
print("="*80)

# Load data (same way as auto_hyperparameter_FIXED.py)
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:10]

all_data = []
for f in files:
    df = pd.read_pickle(f)
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
graphs = combined['input'].tolist()
labels = combined['target'].tolist()

print(f"Total samples: {len(graphs)}")
print(f"Label distribution: Class 0={labels.count(0)}, Class 1={labels.count(1)}")

# Test with shuffle=True
print("\n" + "="*80)
print("WITH SHUFFLE=TRUE")
print("="*80)

loader_shuffled = DataLoader(graphs, batch_size=16, shuffle=True)
shuffled_stats = []

for batch_idx, batch in enumerate(loader_shuffled):
    if batch_idx >= 5:  # Check first 5 batches
        break
    
    # batch is a PyG Batch object, extract labels from batch.y
    batch_labels = batch.y.squeeze().tolist()
    class_0 = batch_labels.count(0.0)
    class_1 = batch_labels.count(1.0)
    shuffled_stats.append((class_0, class_1))
    
    print(f"Batch {batch_idx}: {batch_labels[:8]}... (0s={class_0}, 1s={class_1})")

# Test with shuffle=False
print("\n" + "="*80)
print("WITH SHUFFLE=FALSE (showing the problem)")
print("="*80)

loader_no_shuffle = DataLoader(graphs, batch_size=16, shuffle=False)
no_shuffle_stats = []

for batch_idx, batch in enumerate(loader_no_shuffle):
    if batch_idx >= 5:
        break
    
    # batch is a PyG Batch object, extract labels from batch.y
    batch_labels = batch.y.squeeze().tolist()
    class_0 = batch_labels.count(0.0)
    class_1 = batch_labels.count(1.0)
    no_shuffle_stats.append((class_0, class_1))
    
    print(f"Batch {batch_idx}: {batch_labels[:8]}... (0s={class_0}, 1s={class_1})")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("✓ Good batches: Mix of 0s and 1s (e.g., 8 zeros, 8 ones)")
print("✗ Bad batches: All 0s or all 1s (model can't learn)")

# Analyze shuffled batches
print(f"\n📊 SHUFFLED BATCHES ANALYSIS:")
shuffled_mixed = sum(1 for c0, c1 in shuffled_stats if c0 > 0 and c1 > 0)
print(f"   Mixed batches (both classes): {shuffled_mixed}/5")
print(f"   Pure batches (single class): {5 - shuffled_mixed}/5")

# Analyze non-shuffled batches
print(f"\n📊 NON-SHUFFLED BATCHES ANALYSIS:")
no_shuffle_mixed = sum(1 for c0, c1 in no_shuffle_stats if c0 > 0 and c1 > 0)
print(f"   Mixed batches (both classes): {no_shuffle_mixed}/5")
print(f"   Pure batches (single class): {5 - no_shuffle_mixed}/5")

# Calculate batch diversity scores
def diversity_score(stats):
    """Calculate how diverse the batches are (0=all same class, 1=perfectly mixed)"""
    scores = []
    for c0, c1 in stats:
        total = c0 + c1
        if total == 0:
            continue
        # Calculate how close to 50/50 split
        ratio = min(c0, c1) / total
        scores.append(ratio)
    return sum(scores) / len(scores) if scores else 0

shuffled_diversity = diversity_score(shuffled_stats)
no_shuffle_diversity = diversity_score(no_shuffle_stats)

print(f"\n🎯 DIVERSITY SCORES (0=bad, 0.5=perfect):")
print(f"   Shuffled batches: {shuffled_diversity:.3f}")
print(f"   Non-shuffled batches: {no_shuffle_diversity:.3f}")

if shuffled_diversity > no_shuffle_diversity:
    print(f"   ✅ Shuffling improves batch diversity by {(shuffled_diversity - no_shuffle_diversity):.3f}")
else:
    print(f"   ⚠️ Shuffling doesn't improve diversity much")

# Check for homogeneous batches (all same class)
print(f"\n🚨 HOMOGENEOUS BATCH CHECK:")
shuffled_homo = sum(1 for c0, c1 in shuffled_stats if c0 == 0 or c1 == 0)
no_shuffle_homo = sum(1 for c0, c1 in no_shuffle_stats if c0 == 0 or c1 == 0)

print(f"   Shuffled - Homogeneous batches: {shuffled_homo}/5")
print(f"   Non-shuffled - Homogeneous batches: {no_shuffle_homo}/5")

if no_shuffle_homo > shuffled_homo:
    print(f"   ✅ Shuffling reduces homogeneous batches by {no_shuffle_homo - shuffled_homo}")
    print(f"   🎯 This explains why shuffle=False causes poor training!")
else:
    print(f"   ⚠️ Both have similar homogeneity")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

if shuffled_diversity > 0.3 and no_shuffle_diversity < 0.2:
    print("🎯 SHUFFLE BUG CONFIRMED!")
    print("   - shuffle=True creates mixed batches (good for learning)")
    print("   - shuffle=False creates homogeneous batches (bad for learning)")
    print("   - This explains the performance difference!")
elif shuffled_diversity > no_shuffle_diversity:
    print("✅ Shuffling helps but may not be the main issue")
else:
    print("⚠️ Shuffling doesn't seem to be the main problem")