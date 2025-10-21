"""
Graph Structure vs Node Features Signal Analysis
Determines what contributes to the model's 55.56% accuracy
"""

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import networkx as nx

print("="*80)
print("GRAPH STRUCTURE vs NODE FEATURES SIGNAL ANALYSIS")
print("="*80)

# Load data
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:4]  # First 4 files

class_0_graphs = []  # Non-vulnerable
class_1_graphs = []  # Vulnerable

print(f"Loading data from {len(files)} files...")

for file_idx, f in enumerate(files):
    df = pd.read_pickle(f)
    print(f"File {file_idx}: {f.name} - {len(df)} graphs")
    
    for idx in range(len(df)):
        graph = df.iloc[idx]['input']
        target = df.iloc[idx]['target']
        
        if target == 0 and len(class_0_graphs) < 30:
            class_0_graphs.append((graph, target))
        elif target == 1 and len(class_1_graphs) < 30:
            class_1_graphs.append((graph, target))
        
        # Stop when we have enough samples
        if len(class_0_graphs) >= 30 and len(class_1_graphs) >= 30:
            break
    
    if len(class_0_graphs) >= 30 and len(class_1_graphs) >= 30:
        break

all_graphs = class_0_graphs + class_1_graphs
print(f"\nData collection complete:")
print(f"  Class 0 (non-vulnerable): {len(class_0_graphs)} graphs")
print(f"  Class 1 (vulnerable): {len(class_1_graphs)} graphs")
print(f"  Total: {len(all_graphs)} graphs")

# Feature extraction functions
def extract_structural_features(graph):
    """Extract graph-level structural features"""
    num_nodes = graph.num_nodes
    num_edges = graph.num_edges
    
    # Basic metrics
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    # Convert to NetworkX for advanced metrics
    edge_index = graph.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    if edge_index.shape[1] > 0:
        edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
    
    # Advanced structural features
    try:
        connected_components = nx.number_connected_components(G)
        largest_cc_size = len(max(nx.connected_components(G), key=len)) if num_nodes > 0 else 0
        avg_clustering = nx.average_clustering(G) if num_nodes > 0 else 0
        
        # Centrality statistics
        if num_nodes > 0 and num_edges > 0:
            degree_centrality = list(nx.degree_centrality(G).values())
            max_degree_centrality = max(degree_centrality) if degree_centrality else 0
            avg_degree_centrality = np.mean(degree_centrality) if degree_centrality else 0
            
            # Betweenness centrality (expensive, so sample for large graphs)
            if num_nodes <= 100:
                betweenness = list(nx.betweenness_centrality(G).values())
                max_betweenness = max(betweenness) if betweenness else 0
                avg_betweenness = np.mean(betweenness) if betweenness else 0
            else:
                # Sample for large graphs
                sample_nodes = np.random.choice(num_nodes, min(50, num_nodes), replace=False)
                betweenness = nx.betweenness_centrality(G, k=sample_nodes)
                betweenness_values = list(betweenness.values())
                max_betweenness = max(betweenness_values) if betweenness_values else 0
                avg_betweenness = np.mean(betweenness_values) if betweenness_values else 0
        else:
            max_degree_centrality = avg_degree_centrality = 0
            max_betweenness = avg_betweenness = 0
            
    except Exception as e:
        # Fallback values if NetworkX operations fail
        connected_components = 1
        largest_cc_size = num_nodes
        avg_clustering = 0
        max_degree_centrality = avg_degree_centrality = 0
        max_betweenness = avg_betweenness = 0
    
    return np.array([
        num_nodes,
        num_edges,
        avg_degree,
        density,
        connected_components,
        largest_cc_size,
        avg_clustering,
        max_degree_centrality,
        avg_degree_centrality,
        max_betweenness,
        avg_betweenness
    ])

def extract_node_features_sample(graph, sample_size=10):
    """Extract a sample of node features"""
    node_features = graph.x.numpy()  # [num_nodes, feature_dim]
    
    # Sample random nodes
    num_nodes = node_features.shape[0]
    if num_nodes <= sample_size:
        sampled_features = node_features
    else:
        indices = np.random.choice(num_nodes, sample_size, replace=False)
        sampled_features = node_features[indices]
    
    # Flatten and pad/truncate to fixed size
    flattened = sampled_features.flatten()
    target_size = sample_size * node_features.shape[1]  # sample_size * feature_dim
    
    if len(flattened) < target_size:
        # Pad with zeros
        padded = np.zeros(target_size)
        padded[:len(flattened)] = flattened
        return padded
    else:
        # Truncate
        return flattened[:target_size]

def extract_pooled_node_features(graph):
    """Extract pooled node features (mean and max)"""
    node_features = graph.x.numpy()  # [num_nodes, feature_dim]
    
    mean_features = np.mean(node_features, axis=0)
    max_features = np.max(node_features, axis=0)
    
    return np.concatenate([mean_features, max_features])

# Extract all feature types
print(f"\n" + "="*80)
print("EXTRACTING FEATURES")
print("="*80)

structural_features = []
node_sample_features = []
pooled_node_features = []
labels = []

print("Extracting features from graphs...")
for i, (graph, target) in enumerate(all_graphs):
    if i % 10 == 0:
        print(f"  Processing graph {i+1}/{len(all_graphs)}")
    
    # Extract different feature types
    struct_feat = extract_structural_features(graph)
    node_sample_feat = extract_node_features_sample(graph, sample_size=5)  # Small sample
    pooled_feat = extract_pooled_node_features(graph)
    
    structural_features.append(struct_feat)
    node_sample_features.append(node_sample_feat)
    pooled_node_features.append(pooled_feat)
    labels.append(target)

# Convert to arrays
structural_features = np.array(structural_features)
node_sample_features = np.array(node_sample_features)
pooled_node_features = np.array(pooled_node_features)
labels = np.array(labels)

print(f"\nFeature shapes:")
print(f"  Structural features: {structural_features.shape}")
print(f"  Node sample features: {node_sample_features.shape}")
print(f"  Pooled node features: {pooled_node_features.shape}")
print(f"  Labels: {labels.shape}")

# Feature names for interpretation
structural_feature_names = [
    'num_nodes', 'num_edges', 'avg_degree', 'density',
    'connected_components', 'largest_cc_size', 'avg_clustering',
    'max_degree_centrality', 'avg_degree_centrality',
    'max_betweenness', 'avg_betweenness'
]

print(f"\nStructural feature statistics:")
for i, name in enumerate(structural_feature_names):
    values = structural_features[:, i]
    print(f"  {name}: mean={values.mean():.4f}, std={values.std():.4f}, range=[{values.min():.4f}, {values.max():.4f}]")

# Train-test split
X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    structural_features, labels, test_size=0.3, random_state=42, stratify=labels
)

X_node_train, X_node_test, _, _ = train_test_split(
    node_sample_features, labels, test_size=0.3, random_state=42, stratify=labels
)

X_pooled_train, X_pooled_test, _, _ = train_test_split(
    pooled_node_features, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"\nTrain/test split:")
print(f"  Train: {len(y_train)} samples")
print(f"  Test: {len(y_test)} samples")
print(f"  Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

# Standardize features
scaler_struct = StandardScaler()
scaler_node = StandardScaler()
scaler_pooled = StandardScaler()

X_struct_train_scaled = scaler_struct.fit_transform(X_struct_train)
X_struct_test_scaled = scaler_struct.transform(X_struct_test)

X_node_train_scaled = scaler_node.fit_transform(X_node_train)
X_node_test_scaled = scaler_node.transform(X_node_test)

X_pooled_train_scaled = scaler_pooled.fit_transform(X_pooled_train)
X_pooled_test_scaled = scaler_pooled.transform(X_pooled_test)

# Train classifiers
print(f"\n" + "="*80)
print("TRAINING CLASSIFIERS")
print("="*80)

results = {}

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_name):
    """Train and evaluate classifiers"""
    print(f"\n{feature_name}:")
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    print(f"  Logistic Regression: {lr_acc:.4f} ({lr_acc*100:.1f}%)")
    print(f"  Random Forest: {rf_acc:.4f} ({rf_acc*100:.1f}%)")
    
    # Feature importance for Random Forest
    if hasattr(rf, 'feature_importances_') and feature_name == "Structural Features":
        print(f"  Top 5 important structural features:")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(5, len(indices))):
            idx = indices[i]
            if idx < len(structural_feature_names):
                print(f"    {i+1}. {structural_feature_names[idx]}: {importances[idx]:.4f}")
    
    return {
        'lr_acc': lr_acc,
        'rf_acc': rf_acc,
        'lr_model': lr,
        'rf_model': rf
    }

# 1. Structural features only
results['structural'] = train_and_evaluate(
    X_struct_train_scaled, X_struct_test_scaled, y_train, y_test,
    "Structural Features Only"
)

# 2. Node sample features only
results['node_sample'] = train_and_evaluate(
    X_node_train_scaled, X_node_test_scaled, y_train, y_test,
    "Node Sample Features Only"
)

# 3. Pooled node features only
results['pooled_node'] = train_and_evaluate(
    X_pooled_train_scaled, X_pooled_test_scaled, y_train, y_test,
    "Pooled Node Features Only"
)

# 4. Combined features
X_combined_train = np.concatenate([X_struct_train_scaled, X_pooled_train_scaled], axis=1)
X_combined_test = np.concatenate([X_struct_test_scaled, X_pooled_test_scaled], axis=1)

results['combined'] = train_and_evaluate(
    X_combined_train, X_combined_test, y_train, y_test,
    "Structural + Pooled Node Features"
)

# Summary and analysis
print(f"\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print(f"{'Feature Type':<25} {'LogReg Acc':<12} {'RF Acc':<12} {'Best':<12}")
print("-" * 65)

best_overall = 0
best_method = ""

for method, result in results.items():
    method_name = {
        'structural': 'Structural Only',
        'node_sample': 'Node Sample Only', 
        'pooled_node': 'Pooled Node Only',
        'combined': 'Structural + Node'
    }[method]
    
    lr_acc = result['lr_acc']
    rf_acc = result['rf_acc']
    best_acc = max(lr_acc, rf_acc)
    best_classifier = 'LogReg' if lr_acc > rf_acc else 'RF'
    
    print(f"{method_name:<25} {lr_acc:<12.4f} {rf_acc:<12.4f} {best_acc:.4f} ({best_classifier})")
    
    if best_acc > best_overall:
        best_overall = best_acc
        best_method = method_name

print(f"\n🏆 BEST PERFORMANCE: {best_method}")
print(f"   Accuracy: {best_overall:.4f} ({best_overall*100:.1f}%)")

# Analysis and insights
print(f"\n" + "="*80)
print("ANALYSIS")
print("="*80)

struct_acc = max(results['structural']['lr_acc'], results['structural']['rf_acc'])
node_acc = max(results['pooled_node']['lr_acc'], results['pooled_node']['rf_acc'])
combined_acc = max(results['combined']['lr_acc'], results['combined']['rf_acc'])

print(f"Key findings:")

# Compare to random baseline (50%)
baseline = 0.5
print(f"  Baseline (random): {baseline:.4f} ({baseline*100:.1f}%)")

if struct_acc > baseline + 0.05:
    print(f"  ✓ Structural features ARE discriminative: {struct_acc:.4f} ({struct_acc*100:.1f}%)")
else:
    print(f"  ❌ Structural features are NOT discriminative: {struct_acc:.4f} ({struct_acc*100:.1f}%)")

if node_acc > baseline + 0.05:
    print(f"  ✓ Node features ARE discriminative: {node_acc:.4f} ({node_acc*100:.1f}%)")
else:
    print(f"  ❌ Node features are NOT discriminative: {node_acc:.4f} ({node_acc*100:.1f}%)")

# Compare to GNN performance (55.56%)
gnn_acc = 0.5556
print(f"\nComparison to GNN performance ({gnn_acc:.4f}):")

if struct_acc >= gnn_acc - 0.02:
    print(f"  🎯 Structural features alone explain GNN performance!")
    print(f"     Structure: {struct_acc:.4f} vs GNN: {gnn_acc:.4f}")
elif combined_acc >= gnn_acc - 0.02:
    print(f"  🎯 Combined features explain GNN performance!")
    print(f"     Combined: {combined_acc:.4f} vs GNN: {gnn_acc:.4f}")
else:
    print(f"  ⚠️ Simple classifiers underperform GNN")
    print(f"     Best: {best_overall:.4f} vs GNN: {gnn_acc:.4f}")
    print(f"     GNN likely uses complex feature interactions")

# Insights about what the GNN is learning
print(f"\n💡 INSIGHTS ABOUT GNN LEARNING:")

if struct_acc > node_acc + 0.05:
    print("   GNN primarily learns from graph structure, not node content")
    print("   This explains why it works despite poor node features")
elif node_acc > struct_acc + 0.05:
    print("   GNN primarily learns from node features")
    print("   Our node feature analysis might have missed something")
else:
    print("   GNN learns from both structure and node features")
    print("   Neither alone is sufficient")

# Recommendations
print(f"\n🔧 RECOMMENDATIONS:")

if struct_acc > 0.6:
    print("   ✓ Focus on improving structural feature extraction")
    print("   ✓ Add more sophisticated graph-level features")
    print("   ✓ Consider graph kernels or structural embeddings")
elif node_acc > 0.6:
    print("   ✓ Focus on improving node feature quality")
    print("   ✓ Better embeddings will have the highest impact")
elif combined_acc > max(struct_acc, node_acc) + 0.05:
    print("   ✓ Both structure and node features contribute")
    print("   ✓ Improve both aspects for best results")
else:
    print("   ⚠️ Neither structural nor node features are strongly discriminative")
    print("   ⚠️ May need fundamentally different approach")
    print("   ⚠️ Consider: different graph representations, external features")

print(f"\n📊 CONCLUSION:")
if best_overall >= gnn_acc - 0.02:
    print(f"✅ Simple classifiers match GNN performance")
    print(f"   This suggests the task is learnable with current features")
    print(f"   GNN architecture may be overkill for this problem")
else:
    print(f"❌ Simple classifiers underperform GNN")
    print(f"   GNN captures complex patterns that linear models miss")
    print(f"   Graph neural networks are necessary for this task")