"""
Graph Pooling Effect Analysis
Compares how different pooling strategies affect class discrimination
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

print("="*80)
print("GRAPH POOLING EFFECT ANALYSIS")
print("="*80)

# Load data
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:3]  # First 3 files

class_0_graphs = []  # Non-vulnerable
class_1_graphs = []  # Vulnerable

print(f"Loading data from {len(files)} files...")

for file_idx, f in enumerate(files):
    df = pd.read_pickle(f)
    print(f"File {file_idx}: {f.name} - {len(df)} graphs")
    
    for idx in range(len(df)):
        graph = df.iloc[idx]['input']
        target = df.iloc[idx]['target']
        
        if target == 0 and len(class_0_graphs) < 20:
            class_0_graphs.append(graph)
        elif target == 1 and len(class_1_graphs) < 20:
            class_1_graphs.append(graph)
        
        # Stop when we have enough samples
        if len(class_0_graphs) >= 20 and len(class_1_graphs) >= 20:
            break
    
    if len(class_0_graphs) >= 20 and len(class_1_graphs) >= 20:
        break

print(f"\nData collection complete:")
print(f"  Class 0 (non-vulnerable): {len(class_0_graphs)} graphs")
print(f"  Class 1 (vulnerable): {len(class_1_graphs)} graphs")

# Extract features for each representation type
print(f"\n" + "="*80)
print("EXTRACTING FEATURES")
print("="*80)

def extract_graph_features(graphs, class_name):
    """Extract node features and different pooled representations"""
    all_node_features = []
    mean_pooled_features = []
    max_pooled_features = []
    
    for i, graph in enumerate(graphs):
        node_features = graph.x  # [num_nodes, feature_dim]
        
        # Collect all individual node features
        for node_idx in range(node_features.shape[0]):
            all_node_features.append(node_features[node_idx].numpy())
        
        # Mean pooling
        mean_pooled = torch.mean(node_features, dim=0).numpy()
        mean_pooled_features.append(mean_pooled)
        
        # Max pooling
        max_pooled = torch.max(node_features, dim=0)[0].numpy()
        max_pooled_features.append(max_pooled)
    
    print(f"{class_name}:")
    print(f"  Individual nodes: {len(all_node_features)}")
    print(f"  Mean pooled graphs: {len(mean_pooled_features)}")
    print(f"  Max pooled graphs: {len(max_pooled_features)}")
    
    return (np.array(all_node_features), 
            np.array(mean_pooled_features), 
            np.array(max_pooled_features))

# Extract features for both classes
class_0_nodes, class_0_mean, class_0_max = extract_graph_features(class_0_graphs, "Class 0")
class_1_nodes, class_1_mean, class_1_max = extract_graph_features(class_1_graphs, "Class 1")

# Analysis functions
def compute_distances(features_a, features_b, n_samples=500):
    """Compute within and between class distances"""
    random.seed(42)
    
    # Within class A distances
    within_a = []
    if len(features_a) > 1:
        for _ in range(min(n_samples, len(features_a)*(len(features_a)-1)//2)):
            idx1, idx2 = random.sample(range(len(features_a)), 2)
            dist = np.linalg.norm(features_a[idx1] - features_a[idx2])
            within_a.append(dist)
    
    # Within class B distances
    within_b = []
    if len(features_b) > 1:
        for _ in range(min(n_samples, len(features_b)*(len(features_b)-1)//2)):
            idx1, idx2 = random.sample(range(len(features_b)), 2)
            dist = np.linalg.norm(features_b[idx1] - features_b[idx2])
            within_b.append(dist)
    
    # Between class distances
    between = []
    for _ in range(min(n_samples, len(features_a)*len(features_b))):
        idx_a = random.randint(0, len(features_a) - 1)
        idx_b = random.randint(0, len(features_b) - 1)
        dist = np.linalg.norm(features_a[idx_a] - features_b[idx_b])
        between.append(dist)
    
    within_a = np.array(within_a)
    within_b = np.array(within_b)
    between = np.array(between)
    
    # Combine within-class distances
    within_all = np.concatenate([within_a, within_b]) if len(within_a) > 0 and len(within_b) > 0 else np.array([])
    
    return within_all, between

def analyze_separation(features_0, features_1, representation_name):
    """Analyze class separation for a given representation"""
    print(f"\n{representation_name}:")
    print(f"  Class 0 shape: {features_0.shape}")
    print(f"  Class 1 shape: {features_1.shape}")
    
    # Compute class means and differences
    mean_0 = features_0.mean(axis=0)
    mean_1 = features_1.mean(axis=0)
    difference = np.abs(mean_0 - mean_1)
    
    print(f"  Mean absolute difference: {difference.mean():.6f}")
    print(f"  Max difference: {difference.max():.6f}")
    
    # Compute distances
    within_distances, between_distances = compute_distances(features_0, features_1)
    
    if len(within_distances) > 0 and len(between_distances) > 0:
        within_mean = within_distances.mean()
        between_mean = between_distances.mean()
        separation_ratio = between_mean / within_mean if within_mean > 0 else 0
        
        print(f"  Within-class distance: {within_mean:.4f} ± {within_distances.std():.4f}")
        print(f"  Between-class distance: {between_mean:.4f} ± {between_distances.std():.4f}")
        print(f"  Separation ratio: {separation_ratio:.4f}")
        
        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(between_distances, within_distances)
        significance = "✓ SIGNIFICANT" if p_value < 0.05 else "✗ Not significant"
        print(f"  Statistical test: t={t_stat:.3f}, p={p_value:.6f} {significance}")
        
        return {
            'mean_diff': difference.mean(),
            'max_diff': difference.max(),
            'within_dist': within_mean,
            'between_dist': between_mean,
            'separation_ratio': separation_ratio,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    else:
        print(f"  ⚠️ Cannot compute distances (insufficient data)")
        return {
            'mean_diff': difference.mean(),
            'max_diff': difference.max(),
            'within_dist': 0,
            'between_dist': 0,
            'separation_ratio': 0,
            'p_value': 1.0,
            'significant': False
        }

# Perform analysis for each representation
print(f"\n" + "="*80)
print("CLASS SEPARATION ANALYSIS")
print("="*80)

results = {}

# 1. Individual node features (before pooling)
results['nodes'] = analyze_separation(class_0_nodes, class_1_nodes, "INDIVIDUAL NODE FEATURES")

# 2. Mean pooled features
results['mean'] = analyze_separation(class_0_mean, class_1_mean, "MEAN POOLED FEATURES")

# 3. Max pooled features
results['max'] = analyze_separation(class_0_max, class_1_max, "MAX POOLED FEATURES")

# Comparison and recommendations
print(f"\n" + "="*80)
print("POOLING STRATEGY COMPARISON")
print("="*80)

print("Summary of separation metrics:")
print(f"{'Strategy':<20} {'Mean Diff':<12} {'Sep Ratio':<12} {'Significant':<12}")
print("-" * 60)

for strategy, result in results.items():
    strategy_name = {
        'nodes': 'Individual Nodes',
        'mean': 'Mean Pooling', 
        'max': 'Max Pooling'
    }[strategy]
    
    print(f"{strategy_name:<20} {result['mean_diff']:<12.6f} {result['separation_ratio']:<12.4f} {'Yes' if result['significant'] else 'No':<12}")

# Find best strategy
best_strategy = max(results.keys(), key=lambda k: results[k]['separation_ratio'])
best_result = results[best_strategy]

print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
print(f"   Separation ratio: {best_result['separation_ratio']:.4f}")
print(f"   Statistical significance: {'Yes' if best_result['significant'] else 'No'}")

# Detailed analysis
print(f"\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Check if pooling helps or hurts
node_ratio = results['nodes']['separation_ratio']
mean_ratio = results['mean']['separation_ratio']
max_ratio = results['max']['separation_ratio']

print("Effect of pooling on discrimination:")

if mean_ratio > node_ratio:
    improvement = ((mean_ratio - node_ratio) / node_ratio) * 100 if node_ratio > 0 else float('inf')
    print(f"  ✓ Mean pooling IMPROVES separation by {improvement:.1f}%")
else:
    degradation = ((node_ratio - mean_ratio) / node_ratio) * 100 if node_ratio > 0 else 0
    print(f"  ❌ Mean pooling DEGRADES separation by {degradation:.1f}%")

if max_ratio > node_ratio:
    improvement = ((max_ratio - node_ratio) / node_ratio) * 100 if node_ratio > 0 else float('inf')
    print(f"  ✓ Max pooling IMPROVES separation by {improvement:.1f}%")
else:
    degradation = ((node_ratio - max_ratio) / node_ratio) * 100 if node_ratio > 0 else 0
    print(f"  ❌ Max pooling DEGRADES separation by {degradation:.1f}%")

# Recommendations
print(f"\n🎯 RECOMMENDATIONS:")

if best_result['separation_ratio'] < 1.05:  # Very poor separation
    print("❌ ALL POOLING STRATEGIES SHOW POOR DISCRIMINATION")
    print("   Root cause: Node-level features lack discriminative power")
    print("   Solution: Fix node-level feature extraction first")
    
elif best_strategy == 'nodes':
    print("⚠️ INDIVIDUAL NODES ARE MOST DISCRIMINATIVE")
    print("   Pooling is destroying what little signal exists")
    print("   Consider: Attention-based pooling, hierarchical pooling")
    
elif best_strategy == 'mean':
    print("✓ MEAN POOLING IS OPTIMAL")
    print("   Current model choice is appropriate")
    print("   Consider: Weighted mean pooling, learnable pooling")
    
elif best_strategy == 'max':
    print("✓ MAX POOLING IS OPTIMAL")
    print("   Consider switching from mean to max pooling")
    print("   Max pooling preserves discriminative features better")

# Additional insights
print(f"\n💡 INSIGHTS:")

if results['max']['separation_ratio'] > results['mean']['separation_ratio']:
    print("   Max pooling > Mean pooling suggests discriminative features are sparse")
    print("   Most nodes are non-discriminative, but some nodes carry strong signals")
else:
    print("   Mean pooling ≥ Max pooling suggests discriminative features are distributed")
    print("   Signal is spread across many nodes rather than concentrated")

if all(not result['significant'] for result in results.values()):
    print("   No pooling strategy achieves statistical significance")
    print("   This confirms the node-level feature quality issue")
else:
    significant_strategies = [k for k, v in results.items() if v['significant']]
    print(f"   Significant strategies: {', '.join(significant_strategies)}")
    print("   These strategies preserve some discriminative signal")

print(f"\n📊 CONCLUSION:")
if best_result['separation_ratio'] > 1.1 and best_result['significant']:
    print(f"✅ {best_strategy.upper()} pooling provides meaningful class separation")
    print("   Pooling strategy optimization could improve model performance")
else:
    print("❌ No pooling strategy provides strong class separation")
    print("   Confirms that node-level features are the primary bottleneck")
    print("   Focus on improving node feature quality rather than pooling strategy")