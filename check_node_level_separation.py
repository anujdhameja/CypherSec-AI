"""
Node-Level Class Separation Analysis
Determines if individual nodes can discriminate between vulnerable/non-vulnerable code
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats
import random

print("="*80)
print("NODE-LEVEL CLASS SEPARATION ANALYSIS")
print("="*80)

# Load data
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:5]  # First 5 files

class_0_nodes = []  # Non-vulnerable
class_1_nodes = []  # Vulnerable
graph_count = {0: 0, 1: 0}

print(f"Loading data from {len(files)} files...")

for file_idx, f in enumerate(files):
    df = pd.read_pickle(f)
    print(f"File {file_idx}: {f.name} - {len(df)} graphs")
    
    for idx in range(min(10, len(df))):  # First 10 graphs per file
        graph = df.iloc[idx]['input']
        target = df.iloc[idx]['target']
        
        # Extract all node features from this graph
        node_features = graph.x  # Shape: [num_nodes, feature_dim]
        
        # Add each node to the appropriate class
        for node_idx in range(node_features.shape[0]):
            node_feature = node_features[node_idx].numpy()
            
            if target == 0:
                class_0_nodes.append(node_feature)
            else:
                class_1_nodes.append(node_feature)
        
        graph_count[target] += 1

print(f"\nData collection complete:")
print(f"  Class 0 (non-vulnerable): {len(class_0_nodes)} nodes from {graph_count[0]} graphs")
print(f"  Class 1 (vulnerable): {len(class_1_nodes)} nodes from {graph_count[1]} graphs")

# Convert to numpy arrays
class_0_features = np.array(class_0_nodes)  # [N0, feature_dim]
class_1_features = np.array(class_1_nodes)  # [N1, feature_dim]

print(f"  Class 0 features shape: {class_0_features.shape}")
print(f"  Class 1 features shape: {class_1_features.shape}")

# 1. Compute class statistics
print(f"\n" + "="*80)
print("CLASS STATISTICS")
print("="*80)

class_0_mean = class_0_features.mean(axis=0)
class_1_mean = class_1_features.mean(axis=0)
class_0_std = class_0_features.std(axis=0)
class_1_std = class_1_features.std(axis=0)

difference = np.abs(class_0_mean - class_1_mean)

print(f"Class 0 (non-vulnerable) nodes:")
print(f"  Mean range: [{class_0_mean.min():.4f}, {class_0_mean.max():.4f}]")
print(f"  Mean avg: {class_0_mean.mean():.4f}")
print(f"  Std avg: {class_0_std.mean():.4f}")

print(f"\nClass 1 (vulnerable) nodes:")
print(f"  Mean range: [{class_1_mean.min():.4f}, {class_1_mean.max():.4f}]")
print(f"  Mean avg: {class_1_mean.mean():.4f}")
print(f"  Std avg: {class_1_std.mean():.4f}")

print(f"\nClass differences:")
print(f"  Absolute difference range: [{difference.min():.4f}, {difference.max():.4f}]")
print(f"  Average absolute difference: {difference.mean():.4f}")

# 2. Statistical significance test
print(f"\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

# Test each feature dimension
significant_features = 0
p_values = []

print("Testing each feature dimension (first 10 shown):")
for dim in range(min(10, class_0_features.shape[1])):
    class_0_dim = class_0_features[:, dim]
    class_1_dim = class_1_features[:, dim]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(class_0_dim, class_1_dim)
    p_values.append(p_value)
    
    if p_value < 0.05:
        significant_features += 1
        significance = "✓ SIGNIFICANT"
    else:
        significance = "✗ Not significant"
    
    print(f"  Dim {dim:2d}: t={t_stat:6.3f}, p={p_value:.6f} {significance}")

# Overall significance
all_p_values = np.array(p_values)
significant_ratio = significant_features / len(p_values)

print(f"\nOverall significance:")
print(f"  Significant dimensions: {significant_features}/{len(p_values)} ({significant_ratio:.1%})")
print(f"  Mean p-value: {all_p_values.mean():.6f}")

# 3. Distance analysis
print(f"\n" + "="*80)
print("DISTANCE ANALYSIS")
print("="*80)

# Sample random pairs for distance computation
n_samples = 1000
random.seed(42)

# Within-class distances (Class 0)
within_0_distances = []
for _ in range(n_samples):
    idx1, idx2 = random.sample(range(len(class_0_features)), 2)
    dist = np.linalg.norm(class_0_features[idx1] - class_0_features[idx2])
    within_0_distances.append(dist)

# Within-class distances (Class 1)
within_1_distances = []
for _ in range(n_samples):
    idx1, idx2 = random.sample(range(len(class_1_features)), 2)
    dist = np.linalg.norm(class_1_features[idx1] - class_1_features[idx2])
    within_1_distances.append(dist)

# Cross-class distances
cross_distances = []
for _ in range(n_samples):
    idx0 = random.randint(0, len(class_0_features) - 1)
    idx1 = random.randint(0, len(class_1_features) - 1)
    dist = np.linalg.norm(class_0_features[idx0] - class_1_features[idx1])
    cross_distances.append(dist)

within_0_distances = np.array(within_0_distances)
within_1_distances = np.array(within_1_distances)
cross_distances = np.array(cross_distances)

print(f"Distance statistics (based on {n_samples} random pairs):")
print(f"  Within Class 0: mean={within_0_distances.mean():.4f}, std={within_0_distances.std():.4f}")
print(f"  Within Class 1: mean={within_1_distances.mean():.4f}, std={within_1_distances.std():.4f}")
print(f"  Cross-class:    mean={cross_distances.mean():.4f}, std={cross_distances.std():.4f}")

# Test if cross-class distances are significantly larger
within_all = np.concatenate([within_0_distances, within_1_distances])
t_stat_dist, p_value_dist = stats.ttest_ind(cross_distances, within_all)

print(f"\nDistance separation test:")
print(f"  Cross-class vs Within-class: t={t_stat_dist:.3f}, p={p_value_dist:.6f}")
if p_value_dist < 0.05:
    print(f"  ✓ Cross-class distances are significantly larger")
else:
    print(f"  ✗ Cross-class distances are NOT significantly larger")

# 4. Discriminative power analysis
print(f"\n" + "="*80)
print("DISCRIMINATIVE POWER ANALYSIS")
print("="*80)

# Find most discriminative features
top_discriminative_dims = np.argsort(difference)[-10:][::-1]  # Top 10 most different

print("Top 10 most discriminative feature dimensions:")
for i, dim in enumerate(top_discriminative_dims):
    diff = difference[dim]
    c0_mean = class_0_mean[dim]
    c1_mean = class_1_mean[dim]
    print(f"  {i+1:2d}. Dim {dim:2d}: diff={diff:.4f}, C0={c0_mean:.4f}, C1={c1_mean:.4f}")

# Signal-to-noise ratio
signal = difference
noise = (class_0_std + class_1_std) / 2
snr = signal / (noise + 1e-8)  # Add small epsilon to avoid division by zero

print(f"\nSignal-to-Noise Ratio:")
print(f"  Mean SNR: {snr.mean():.4f}")
print(f"  Max SNR: {snr.max():.4f}")
print(f"  SNR > 0.1: {(snr > 0.1).sum()}/{len(snr)} dimensions")

# 5. Conclusion
print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

issues = []
strengths = []

# Check various criteria
if difference.mean() < 0.01:
    issues.append("Very small class differences at node level")
else:
    strengths.append("Reasonable class differences at node level")

if significant_ratio < 0.1:
    issues.append("Very few statistically significant feature dimensions")
elif significant_ratio < 0.3:
    issues.append("Limited statistically significant feature dimensions")
else:
    strengths.append("Many statistically significant feature dimensions")

if p_value_dist >= 0.05:
    issues.append("Cross-class distances not significantly larger than within-class")
else:
    strengths.append("Cross-class distances significantly larger than within-class")

if snr.mean() < 0.1:
    issues.append("Very low signal-to-noise ratio")
elif snr.mean() < 0.3:
    issues.append("Low signal-to-noise ratio")
else:
    strengths.append("Good signal-to-noise ratio")

print("STRENGTHS:")
for strength in strengths:
    print(f"  ✓ {strength}")

print("\nISSUES:")
for issue in issues:
    print(f"  ❌ {issue}")

print(f"\n🎯 DIAGNOSIS:")
if len(issues) > len(strengths):
    print("❌ PROBLEM IS AT NODE LEVEL")
    print("   Individual nodes cannot discriminate between classes")
    print("   This explains why graph-level aggregation also fails")
    print("   Root cause: Node features are not class-discriminative")
    
    print(f"\n🔧 SOLUTIONS:")
    print("   1. Improve node feature extraction (better Word2Vec application)")
    print("   2. Use different node content (more discriminative text)")
    print("   3. Add structural features to node representations")
    print("   4. Use different embedding strategy (e.g., contextualized embeddings)")
    
else:
    print("✅ NODE LEVEL SEPARATION EXISTS")
    print("   Individual nodes can somewhat discriminate between classes")
    print("   Problem is likely in graph-level aggregation")
    print("   Root cause: Pooling strategy destroys discriminative information")
    
    print(f"\n🔧 SOLUTIONS:")
    print("   1. Try different pooling strategies (attention, max, sum)")
    print("   2. Use hierarchical pooling")
    print("   3. Add graph-level features")
    print("   4. Use graph-level attention mechanisms")

print(f"\n📊 KEY METRICS:")
print(f"   Average class difference: {difference.mean():.4f}")
print(f"   Significant dimensions: {significant_ratio:.1%}")
print(f"   Cross-class separation: {'Yes' if p_value_dist < 0.05 else 'No'}")
print(f"   Signal-to-noise ratio: {snr.mean():.4f}")

if difference.mean() > 0.05 and significant_ratio > 0.3:
    print(f"\n🎯 RECOMMENDATION: Focus on improving graph-level aggregation")
else:
    print(f"\n🎯 RECOMMENDATION: Focus on improving node-level features")