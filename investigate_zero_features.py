#!/usr/bin/env python3
"""
Investigate Zero Features
Find corrupted graphs and trace back to source data
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from collections import Counter

def investigate_zero_features():
    """Find and analyze graphs with zero or corrupted features"""
    print("=" * 80)
    print("INVESTIGATING ZERO FEATURES")
    print("=" * 80)
    
    # Load input data files
    input_dir = Path('data/input')
    input_files = list(input_dir.glob('*.pkl'))
    
    if not input_files:
        print("❌ No input files found in data/input/")
        return
    
    print(f"Found {len(input_files)} input files")
    
    zero_feature_graphs = []
    identical_feature_graphs = []
    normal_graphs = []
    all_graphs = []
    
    # Process each file
    for file_idx, file_path in enumerate(input_files[:5]):  # First 5 files for speed
        print(f"\nProcessing {file_path.name}...")
        
        try:
            df = pd.read_pickle(file_path)
            print(f"  Loaded DataFrame with {len(df)} samples")
            
            if 'input' not in df.columns:
                print(f"  ⚠️ No 'input' column found. Columns: {list(df.columns)}")
                continue
            
            # Analyze each graph in this file
            for idx, graph in enumerate(df['input']):
                graph_id = f"{file_path.stem}_{idx}"
                
                try:
                    # Basic graph info
                    num_nodes = graph.x.shape[0]
                    num_edges = graph.edge_index.shape[1]
                    label = graph.y.item()
                    
                    # Feature analysis
                    features = graph.x.numpy()
                    zero_ratio = (features == 0).mean()
                    feature_mean = features.mean()
                    feature_std = features.std()
                    
                    # Check for identical features across nodes
                    if num_nodes > 1:
                        feature_variance = features.var(axis=0).mean()
                        is_identical = feature_variance < 1e-6
                    else:
                        is_identical = False
                    
                    graph_info = {
                        'id': graph_id,
                        'file': file_path.name,
                        'index': idx,
                        'num_nodes': int(num_nodes),
                        'num_edges': int(num_edges),
                        'label': float(label),
                        'zero_ratio': float(zero_ratio),
                        'feature_mean': float(feature_mean),
                        'feature_std': float(feature_std),
                        'is_identical': bool(is_identical),
                        'feature_variance': float(feature_variance if num_nodes > 1 else 0)
                    }
                    
                    all_graphs.append(graph_info)
                    
                    # Categorize
                    if zero_ratio > 0.9:  # >90% zeros
                        zero_feature_graphs.append(graph_info)
                    elif is_identical:
                        identical_feature_graphs.append(graph_info)
                    else:
                        normal_graphs.append(graph_info)
                
                except Exception as e:
                    print(f"  ❌ Error processing graph {idx}: {e}")
        
        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")
    
    # Summary statistics
    print(f"\n" + "=" * 80)
    print("FEATURE CORRUPTION ANALYSIS")
    print("=" * 80)
    
    total_graphs = len(all_graphs)
    print(f"Total graphs analyzed: {total_graphs}")
    print(f"Zero feature graphs: {len(zero_feature_graphs)} ({len(zero_feature_graphs)/total_graphs:.1%})")
    print(f"Identical feature graphs: {len(identical_feature_graphs)} ({len(identical_feature_graphs)/total_graphs:.1%})")
    print(f"Normal graphs: {len(normal_graphs)} ({len(normal_graphs)/total_graphs:.1%})")
    
    # Show examples of corrupted graphs
    print(f"\n🔍 ZERO FEATURE EXAMPLES (first 5):")
    print(f"{'ID':<20} {'Nodes':<8} {'Edges':<8} {'Label':<8} {'Zero%':<8} {'Mean':<10}")
    print("-" * 70)
    
    for graph in zero_feature_graphs[:5]:
        print(f"{graph['id']:<20} {graph['num_nodes']:<8} {graph['num_edges']:<8} "
              f"{graph['label']:<8} {graph['zero_ratio']:<8.1%} {graph['feature_mean']:<10.3f}")
    
    print(f"\n🔍 IDENTICAL FEATURE EXAMPLES (first 5):")
    print(f"{'ID':<20} {'Nodes':<8} {'Edges':<8} {'Label':<8} {'Variance':<12}")
    print("-" * 70)
    
    for graph in identical_feature_graphs[:5]:
        print(f"{graph['id']:<20} {graph['num_nodes']:<8} {graph['num_edges']:<8} "
              f"{graph['label']:<8} {graph['feature_variance']:<12.2e}")
    
    print(f"\n🔍 NORMAL FEATURE EXAMPLES (first 5):")
    print(f"{'ID':<20} {'Nodes':<8} {'Edges':<8} {'Label':<8} {'Mean':<10} {'Std':<10}")
    print("-" * 80)
    
    for graph in normal_graphs[:5]:
        print(f"{graph['id']:<20} {graph['num_nodes']:<8} {graph['num_edges']:<8} "
              f"{graph['label']:<8} {graph['feature_mean']:<10.3f} {graph['feature_std']:<10.3f}")
    
    # Analyze patterns
    print(f"\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)
    
    # Label distribution by corruption type
    zero_labels = [g['label'] for g in zero_feature_graphs]
    identical_labels = [g['label'] for g in identical_feature_graphs]
    normal_labels = [g['label'] for g in normal_graphs]
    
    print(f"Zero feature label distribution: {dict(Counter(zero_labels))}")
    print(f"Identical feature label distribution: {dict(Counter(identical_labels))}")
    print(f"Normal feature label distribution: {dict(Counter(normal_labels))}")
    
    # Size patterns
    zero_sizes = [g['num_nodes'] for g in zero_feature_graphs]
    identical_sizes = [g['num_nodes'] for g in identical_feature_graphs]
    normal_sizes = [g['num_nodes'] for g in normal_graphs]
    
    if zero_sizes:
        print(f"\nZero feature graph sizes: avg={np.mean(zero_sizes):.1f}, "
              f"min={min(zero_sizes)}, max={max(zero_sizes)}")
    if identical_sizes:
        print(f"Identical feature graph sizes: avg={np.mean(identical_sizes):.1f}, "
              f"min={min(identical_sizes)}, max={max(identical_sizes)}")
    if normal_sizes:
        print(f"Normal graph sizes: avg={np.mean(normal_sizes):.1f}, "
              f"min={min(normal_sizes)}, max={max(normal_sizes)}")
    
    # Save detailed results
    results = {
        'summary': {
            'total_graphs': total_graphs,
            'zero_feature_count': len(zero_feature_graphs),
            'identical_feature_count': len(identical_feature_graphs),
            'normal_count': len(normal_graphs)
        },
        'zero_examples': zero_feature_graphs[:10],
        'identical_examples': identical_feature_graphs[:10],
        'normal_examples': normal_graphs[:10]
    }
    
    with open('zero_features_investigation.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: zero_features_investigation.json")
    
    # Conclusions
    print(f"\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    
    corruption_rate = (len(zero_feature_graphs) + len(identical_feature_graphs)) / total_graphs
    
    if corruption_rate > 0.2:  # >20% corrupted
        print(f"🚨 CRITICAL: {corruption_rate:.1%} of features are corrupted!")
        print("   This explains why all models perform at ~51% (random)")
        print("   The feature extraction pipeline is broken")
    elif corruption_rate > 0.05:  # >5% corrupted
        print(f"⚠️  WARNING: {corruption_rate:.1%} of features are corrupted")
        print("   This may impact model performance")
    else:
        print(f"✅ Feature corruption rate is low ({corruption_rate:.1%})")
        print("   The issue may be elsewhere (labels, task difficulty)")
    
    print(f"\n💡 NEXT STEPS:")
    if corruption_rate > 0.1:
        print("1. 🔧 Debug the embedding pipeline:")
        print("   - Check Word2Vec model loading")
        print("   - Verify token extraction from CPG")
        print("   - Debug NodesEmbedding.embed() method")
        print("2. 🔍 Trace specific corrupted examples back to source")
    else:
        print("1. 📊 Features seem OK, investigate:")
        print("   - Label quality and consistency")
        print("   - Task inherent difficulty")
        print("   - Need for domain-specific features")

if __name__ == "__main__":
    investigate_zero_features()