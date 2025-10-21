import pandas as pd
from pathlib import Path
from collections import Counter

print("="*80)
print("NODE CONTENT INSPECTION")
print("="*80)

input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:3]

empty_nodes = 0
non_empty_nodes = 0
all_node_texts = []
node_field_stats = Counter()

print(f"Checking {len(files)} files...")

for file_idx, f in enumerate(files):
    df = pd.read_pickle(f)
    print(f"\nFile {file_idx}: {f.name} - {len(df)} graphs")
    
    for idx in range(min(5, len(df))):  # Check first 5 graphs per file
        graph = df.iloc[idx]['input']
        print(f"  Graph {idx}: {graph.num_nodes} nodes")
        
        # Check what text each node has
        for node_idx in range(min(10, graph.num_nodes)):  # Check first 10 nodes per graph
            # Try to access node data - PyG graphs store node features in x
            # But we need to check if there's actual node objects or just features
            
            # Method 1: Check if graph has node objects
            node = None
            text = None
            found_field = None
            
            # Try different ways to access node data
            try:
                # Check if graph has nodes as objects (unlikely in PyG)
                if hasattr(graph, 'nodes') and callable(graph.nodes):
                    nodes_list = list(graph.nodes())
                    if node_idx < len(nodes_list):
                        node = nodes_list[node_idx]
                elif hasattr(graph, 'nodes') and not callable(graph.nodes):
                    if node_idx < len(graph.nodes):
                        node = graph.nodes[node_idx]
            except:
                pass
            
            # If we have a node object, try to extract text
            if node is not None:
                # Try different field names
                for field in ['text', 'label', 'value', 'name', 'op', 'code', 'type']:
                    try:
                        if hasattr(node, field):
                            text = getattr(node, field)
                            found_field = field
                            break
                        elif isinstance(node, dict) and field in node:
                            text = node[field]
                            found_field = field
                            break
                    except:
                        continue
            
            # If no node object, the graph might only have feature vectors
            if text is None and node is None:
                # This is likely the case - PyG graphs typically only store x (features)
                # The actual node content might be lost during preprocessing
                text = f"<feature_vector_only>"
                found_field = "x_features"
            
            # Record statistics
            if found_field:
                node_field_stats[found_field] += 1
            
            if text is None or text == "" or str(text).strip() == "":
                empty_nodes += 1
            else:
                non_empty_nodes += 1
                all_node_texts.append(str(text))
            
            # Debug: print first few nodes
            if file_idx == 0 and idx == 0 and node_idx < 3:
                print(f"    Node {node_idx}: field='{found_field}', text='{text}'")

print(f"\n" + "="*80)
print("RESULTS")
print("="*80)

total_nodes = empty_nodes + non_empty_nodes
print(f"Empty nodes: {empty_nodes}")
print(f"Non-empty nodes: {non_empty_nodes}")
print(f"Total nodes checked: {total_nodes}")

if total_nodes > 0:
    print(f"Ratio: {empty_nodes / total_nodes * 100:.1f}% empty")
else:
    print("No nodes found!")

print(f"\nNode field statistics:")
for field, count in node_field_stats.most_common():
    print(f"  {field}: {count} nodes")

if all_node_texts:
    print(f"\nSample node texts (first 20):")
    for i, text in enumerate(all_node_texts[:20]):
        print(f"  {i+1:2d}. '{text}'")
    
    print(f"\nMost common node texts:")
    for text, count in Counter(all_node_texts).most_common(10):
        print(f"  '{text}': {count}")
    
    print(f"\nUnique node texts: {len(set(all_node_texts))}")
else:
    print("\n❌ NO NODE TEXTS FOUND!")

# Additional diagnostic: Check graph structure
print(f"\n" + "="*80)
print("GRAPH STRUCTURE DIAGNOSTIC")
print("="*80)

sample_file = files[0]
df = pd.read_pickle(sample_file)
sample_graph = df.iloc[0]['input']

print(f"Sample graph attributes:")
for attr in dir(sample_graph):
    if not attr.startswith('_'):
        try:
            value = getattr(sample_graph, attr)
            if not callable(value):
                print(f"  {attr}: {type(value)} - {str(value)[:100]}...")
        except:
            print(f"  {attr}: <error accessing>")

# Check if there's any way to get original node data
print(f"\nGraph data keys (if dict-like):")
if hasattr(sample_graph, 'keys'):
    try:
        keys = sample_graph.keys()
        print(f"  Keys: {list(keys)}")
    except:
        print("  Could not access keys")

print(f"\n🔍 ANALYSIS:")
if empty_nodes / total_nodes > 0.8:
    print("❌ CRITICAL: >80% of nodes are empty!")
    print("   This explains why features are meaningless")
    print("   The graph preprocessing is losing node content")
elif all_node_texts and all_node_texts[0] == "<feature_vector_only>":
    print("❌ CRITICAL: Nodes only contain feature vectors!")
    print("   Original node text/code content has been lost")
    print("   This explains why Word2Vec embeddings can't be applied properly")
else:
    print("✓ Nodes contain some text content")
    print("   Need to check if the content is meaningful")