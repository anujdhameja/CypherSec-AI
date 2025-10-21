import pandas as pd
import torch
from pathlib import Path

# Load data files
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:2]

print("="*80)
print("LABEL CONSISTENCY CHECK")
print("="*80)

all_graph_labels = []
all_df_labels = []

for file_idx, f in enumerate(files):
    df = pd.read_pickle(f)
    print(f"\nFile {file_idx}: {f.name}")
    
    for i in range(min(25, len(df))):
        graph = df.iloc[i]['input']
        target = df.iloc[i]['target']
        
        graph_y = graph.y.item() if hasattr(graph, 'y') else None
        
        all_graph_labels.append(graph_y)
        all_df_labels.append(target)
        
        match = (graph_y == target)
        symbol = "✓" if match else "✗"
        
        if i < 10:  # Print first 10
            print(f"  Sample {i}: graph.y={graph_y}, target={target} {symbol}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"Total samples checked: {len(all_graph_labels)}")
print(f"Graph.y distribution: {pd.Series(all_graph_labels).value_counts().to_dict()}")
print(f"Target distribution: {pd.Series(all_df_labels).value_counts().to_dict()}")
print(f"Matches: {sum(g==t for g,t in zip(all_graph_labels, all_df_labels))}/{len(all_graph_labels)}")

# Additional checks
print(f"\n🔍 DETAILED ANALYSIS:")
print(f"Graph.y unique values: {sorted(set(all_graph_labels))}")
print(f"Target unique values: {sorted(set(all_df_labels))}")

# Check for type mismatches
graph_types = [type(g).__name__ for g in all_graph_labels[:5]]
target_types = [type(t).__name__ for t in all_df_labels[:5]]
print(f"Graph.y types (first 5): {graph_types}")
print(f"Target types (first 5): {target_types}")

# Check if all labels are the same (stuck prediction indicator)
if len(set(all_graph_labels)) == 1:
    print(f"⚠️ WARNING: All graph.y labels are the same: {all_graph_labels[0]}")
else:
    print(f"✓ Graph.y labels have variety: {len(set(all_graph_labels))} unique values")

if len(set(all_df_labels)) == 1:
    print(f"⚠️ WARNING: All target labels are the same: {all_df_labels[0]}")
else:
    print(f"✓ Target labels have variety: {len(set(all_df_labels))} unique values")

# Check for mismatches
mismatches = [(i, g, t) for i, (g, t) in enumerate(zip(all_graph_labels, all_df_labels)) if g != t]
if mismatches:
    print(f"\n❌ LABEL MISMATCHES FOUND: {len(mismatches)}")
    for i, g, t in mismatches[:5]:  # Show first 5 mismatches
        print(f"  Sample {i}: graph.y={g} != target={t}")
else:
    print(f"\n✅ ALL LABELS MATCH: graph.y == target for all samples")