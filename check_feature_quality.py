import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("FEATURE QUALITY CHECK")
print("="*80)

# Load data
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:5]

class_features = {0: [], 1: []}

for f in files:
    df = pd.read_pickle(f)
    for idx in range(len(df)):
        graph = df.iloc[idx]['input']
        label = df.iloc[idx]['target']
        
        # Get node features
        x = graph.x  # Shape: [num_nodes, feature_dim]
        
        # Aggregate to graph-level (mean pooling)
        graph_features = x.mean(dim=0).numpy()
        class_features[label].append(graph_features)
        
        if len(class_features[0]) >= 50 and len(class_features[1]) >= 50:
            break
    
    if len(class_features[0]) >= 50 and len(class_features[1]) >= 50:
        break

# Convert to arrays
class_0_features = np.array(class_features[0])  # [num_samples, feature_dim]
class_1_features = np.array(class_features[1])

print(f"\nCollected features:")
print(f" Class 0: {class_0_features.shape}")
print(f" Class 1: {class_1_features.shape}")

# Compute statistics
class_0_mean = class_0_features.mean(axis=0)
class_1_mean = class_1_features.mean(axis=0)
difference = np.abs(class_0_mean - class_1_mean)

print(f"\n" + "="*80)
print("FEATURE STATISTICS")
print("="*80)

print(f"Class 0 feature mean: min={class_0_mean.min():.4f}, max={class_0_mean.max():.4f}, avg={class_0_mean.mean():.4f}")
print(f"Class 1 feature mean: min={class_1_mean.min():.4f}, max={class_1_mean.max():.4f}, avg={class_1_mean.mean():.4f}")
print(f"Absolute difference: min={difference.min():.4f}, max={difference.max():.4f}, avg={difference.mean():.4f}")

# Check if features are discriminative
if difference.mean() < 0.01:
    print("\n⚠️ WARNING: Features are nearly identical between classes!")
    print("  Model cannot learn from these features.")
elif difference.mean() < 0.05:
    print("\n⚠️ CAUTION: Features show weak discrimination between classes.")
    print("  Model will struggle to learn.")
else:
    print("\n✓ Features show good discrimination between classes.")

# Show top discriminative features
top_indices = np.argsort(difference)[-5:][::-1]
print(f"\nTop 5 most discriminative features:")
for idx in top_indices:
    print(f"  Feature {idx}: Class0={class_0_mean[idx]:.4f}, Class1={class_1_mean[idx]:.4f}, Diff={difference[idx]:.4f}")

# Additional analysis
print(f"\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Check for zero features
zero_features_0 = np.sum(class_0_mean == 0)
zero_features_1 = np.sum(class_1_mean == 0)
total_features = len(class_0_mean)

print(f"Zero features in Class 0: {zero_features_0}/{total_features} ({zero_features_0/total_features*100:.1f}%)")
print(f"Zero features in Class 1: {zero_features_1}/{total_features} ({zero_features_1/total_features*100:.1f}%)")

if zero_features_0 > total_features * 0.9 or zero_features_1 > total_features * 0.9:
    print("⚠️ WARNING: Most features are zero! This indicates feature extraction problems.")

# Check feature variance within classes
class_0_std = class_0_features.std(axis=0)
class_1_std = class_1_features.std(axis=0)

print(f"\nFeature variance within classes:")
print(f"Class 0 std: min={class_0_std.min():.4f}, max={class_0_std.max():.4f}, avg={class_0_std.mean():.4f}")
print(f"Class 1 std: min={class_1_std.min():.4f}, max={class_1_std.max():.4f}, avg={class_1_std.mean():.4f}")

# Signal-to-noise ratio
signal_to_noise = difference / (class_0_std + class_1_std + 1e-8)  # Add small epsilon to avoid division by zero
avg_snr = signal_to_noise.mean()

print(f"\nSignal-to-Noise Ratio: {avg_snr:.4f}")
if avg_snr < 0.1:
    print("⚠️ WARNING: Very low signal-to-noise ratio! Features are noisy.")
elif avg_snr < 0.5:
    print("⚠️ CAUTION: Low signal-to-noise ratio. Features may be noisy.")
else:
    print("✓ Good signal-to-noise ratio.")

# Check if all features are the same (indicating random embeddings)
all_same_0 = np.all(class_0_features == class_0_features[0], axis=0)
all_same_1 = np.all(class_1_features == class_1_features[0], axis=0)

same_features_0 = np.sum(all_same_0)
same_features_1 = np.sum(all_same_1)

print(f"\nIdentical features across samples:")
print(f"Class 0: {same_features_0}/{total_features} features are identical across all samples")
print(f"Class 1: {same_features_1}/{total_features} features are identical across all samples")

if same_features_0 > total_features * 0.5 or same_features_1 > total_features * 0.5:
    print("⚠️ WARNING: Many features are identical across samples! This suggests random/fixed embeddings.")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Overall assessment
issues = []
if difference.mean() < 0.05:
    issues.append("Poor class discrimination")
if zero_features_0 > total_features * 0.9 or zero_features_1 > total_features * 0.9:
    issues.append("Too many zero features")
if avg_snr < 0.1:
    issues.append("Very low signal-to-noise ratio")
if same_features_0 > total_features * 0.5 or same_features_1 > total_features * 0.5:
    issues.append("Too many identical features")

if not issues:
    print("✅ FEATURES LOOK GOOD!")
    print("   Features are discriminative and should allow the model to learn.")
else:
    print("❌ FEATURE QUALITY ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
    print("\n🔧 RECOMMENDATIONS:")
    if "Poor class discrimination" in issues:
        print("   - Check Word2Vec model quality")
        print("   - Verify node feature extraction process")
    if "Too many zero features" in issues:
        print("   - Check if Word2Vec embeddings are being applied correctly")
        print("   - Verify feature preprocessing pipeline")
    if "Very low signal-to-noise ratio" in issues:
        print("   - Consider feature normalization")
        print("   - Check for feature corruption")
    if "Too many identical features" in issues:
        print("   - Verify that different graphs have different features")
        print("   - Check if random embeddings are being used incorrectly")