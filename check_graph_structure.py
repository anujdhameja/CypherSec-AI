import pandas as pd
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("GRAPH STRUCTURE ANALYSIS")
print("="*80)

# Load data
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:5]

class_stats = {0: defaultdict(list), 1: defaultdict(list)}

for f in files:
    df = pd.read_pickle(f)
    for idx in range(len(df)):
        graph = df.iloc[idx]['input']
        label = df.iloc[idx]['target']
        
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        class_stats[label]['num_nodes'].append(num_nodes)
        class_stats[label]['num_edges'].append(num_edges)
        class_stats[label]['avg_degree'].append(avg_degree)
        class_stats[label]['density'].append(density)
        
        if len(class_stats[0]['num_nodes']) >= 100 and len(class_stats[1]['num_nodes']) >= 100:
            break
    
    if len(class_stats[0]['num_nodes']) >= 100 and len(class_stats[1]['num_nodes']) >= 100:
        break

print(f"Collected samples:")
print(f" Class 0: {len(class_stats[0]['num_nodes'])} graphs")
print(f" Class 1: {len(class_stats[1]['num_nodes'])} graphs")

print(f"\n" + "="*80)
print("STRUCTURE COMPARISON")
print("="*80)

significant_differences = 0
total_metrics = 4

for metric in ['num_nodes', 'num_edges', 'avg_degree', 'density']:
    class_0_vals = np.array(class_stats[0][metric])
    class_1_vals = np.array(class_stats[1][metric])
    
    c0_mean = class_0_vals.mean()
    c1_mean = class_1_vals.mean()
    c0_std = class_0_vals.std()
    c1_std = class_1_vals.std()
    
    percent_diff = abs(c0_mean - c1_mean) / c0_mean * 100 if c0_mean > 0 else 0
    
    print(f"\n{metric.upper()}:")
    print(f"  Class 0: mean={c0_mean:.2f}, std={c0_std:.2f}")
    print(f"  Class 1: mean={c1_mean:.2f}, std={c1_std:.2f}")
    print(f"  Difference: {percent_diff:.1f}%")
    
    if percent_diff < 5:
        print(f"  ⚠️ Classes very similar in {metric}")
    else:
        print(f"  ✓ Classes differ in {metric}")
        significant_differences += 1

# Additional detailed analysis
print(f"\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

# Check for extreme values
for class_label in [0, 1]:
    nodes = np.array(class_stats[class_label]['num_nodes'])
    edges = np.array(class_stats[class_label]['num_edges'])
    
    print(f"\nClass {class_label} detailed stats:")
    print(f"  Nodes: min={nodes.min()}, max={nodes.max()}, median={np.median(nodes):.1f}")
    print(f"  Edges: min={edges.min()}, max={edges.max()}, median={np.median(edges):.1f}")
    
    # Check for very small or very large graphs
    small_graphs = np.sum(nodes < 10)
    large_graphs = np.sum(nodes > 200)
    print(f"  Small graphs (<10 nodes): {small_graphs}/{len(nodes)} ({small_graphs/len(nodes)*100:.1f}%)")
    print(f"  Large graphs (>200 nodes): {large_graphs}/{len(nodes)} ({large_graphs/len(nodes)*100:.1f}%)")

# Statistical significance test (simple t-test approximation)
print(f"\n" + "="*80)
print("STATISTICAL SIGNIFICANCE")
print("="*80)

from scipy import stats

for metric in ['num_nodes', 'num_edges', 'avg_degree', 'density']:
    class_0_vals = np.array(class_stats[0][metric])
    class_1_vals = np.array(class_stats[1][metric])
    
    try:
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(class_0_vals, class_1_vals)
        
        print(f"{metric.upper()}:")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  ✓ Statistically significant difference (p < 0.05)")
        else:
            print(f"  ⚠️ No statistically significant difference (p >= 0.05)")
    except Exception as e:
        print(f"{metric.upper()}: Could not compute t-test ({e})")

print(f"\n" + "="*80)
print("GRAPH COMPLEXITY ANALYSIS")
print("="*80)

# Analyze graph complexity patterns
for class_label in [0, 1]:
    nodes = np.array(class_stats[class_label]['num_nodes'])
    edges = np.array(class_stats[class_label]['num_edges'])
    density = np.array(class_stats[class_label]['density'])
    
    # Calculate complexity score (combination of size and connectivity)
    complexity_score = nodes * density  # Larger, denser graphs are more complex
    
    print(f"\nClass {class_label} complexity:")
    print(f"  Complexity score: mean={complexity_score.mean():.2f}, std={complexity_score.std():.2f}")
    
    # Categorize graphs by complexity
    simple_graphs = np.sum(complexity_score < 5)
    medium_graphs = np.sum((complexity_score >= 5) & (complexity_score < 20))
    complex_graphs = np.sum(complexity_score >= 20)
    
    total = len(complexity_score)
    print(f"  Simple graphs (<5): {simple_graphs}/{total} ({simple_graphs/total*100:.1f}%)")
    print(f"  Medium graphs (5-20): {medium_graphs}/{total} ({medium_graphs/total*100:.1f}%)")
    print(f"  Complex graphs (>20): {complex_graphs}/{total} ({complex_graphs/total*100:.1f}%)")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"Metrics with significant differences: {significant_differences}/{total_metrics}")

if significant_differences == 0:
    print("\n❌ CRITICAL ISSUE: No structural differences between classes!")
    print("   - Graphs have identical structure regardless of vulnerability")
    print("   - GNN cannot learn from structure alone")
    print("   - Model must rely entirely on node features")
    print("\n🔧 IMPLICATIONS:")
    print("   - Node features become critically important")
    print("   - Graph structure provides no discriminative signal")
    print("   - This explains why feature quality is so crucial")
    
elif significant_differences < 2:
    print("\n⚠️ WARNING: Limited structural differences between classes")
    print("   - Some structural patterns exist but are weak")
    print("   - GNN will struggle to use structural information effectively")
    print("   - Model performance will depend heavily on node features")
    
else:
    print("\n✅ GOOD: Classes show structural differences")
    print("   - GNN can potentially learn from graph structure")
    print("   - Structural patterns may help classification")
    print("   - Both structure and features contribute to learning")

print(f"\n🎯 KEY INSIGHT:")
if significant_differences == 0:
    print("   The lack of structural differences explains why the model")
    print("   struggles even with a working training pipeline. Without")
    print("   meaningful node features OR structural differences,")
    print("   the model has no discriminative signal to learn from.")
else:
    print("   Structural differences exist, so the main bottleneck")
    print("   is likely the poor node feature quality we found earlier.")