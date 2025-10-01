#This is just to see how does the cpg looks like 

import json
import networkx as nx

# Load Joern GraphSON
with open("C:/Devign/devign/data/cpg/graph_json/export.json") as f:
    data = json.load(f)

# Create a directed graph
G = nx.DiGraph()

# # Add nodes
# for node in data["@value"]["vertices"]:
#     node_id = node["id"]["@value"]
#     node_label = node.get("label", "NODE")
#     properties = {k: v["@value"] for k, v in node.get("properties", {}).items()}
#     G.add_node(node_id, label=node_label, **properties)

# # Add edges
# for edge in data["@value"]["edges"]:
#     out_id = edge["outV"]["@value"]
#     in_id = edge["inV"]["@value"]
#     label = edge.get("label", "")
#     properties = {k: v["@value"] for k, v in edge.get("properties", {}).items()}
#     G.add_edge(out_id, in_id, label=label, **properties)

# Add nodes
for node in data["@value"]["vertices"]:
    node_id = node["id"]["@value"]
    node_label = node.get("label", "NODE")
    properties = {}
    for k, v in node.get("properties", {}).items():
        # Flatten nested dicts/lists to string
        if isinstance(v, dict) and "@value" in v:
            properties[k] = str(v["@value"])
        else:
            properties[k] = str(v)
    G.add_node(node_id, label=node_label, **properties)

# Add edges
for edge in data["@value"]["edges"]:
    out_id = edge["outV"]["@value"]
    in_id = edge["inV"]["@value"]
    label = edge.get("label", "")
    properties = {}
    for k, v in edge.get("properties", {}).items():
        if isinstance(v, dict) and "@value" in v:
            properties[k] = str(v["@value"])
        else:
            properties[k] = str(v)
    G.add_edge(out_id, in_id, label=label, **properties)


# Export to GraphML
nx.write_graphml(G, "C:/Devign/devign/data/cpg/graph_full.graphml")
print("GraphML export complete!")
