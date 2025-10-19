import re
import numpy as np
import torch
from torch_geometric.data import Data
from src.utils.functions.parse import tokenizer
from src.utils import log as logger
from gensim.models.keyedvectors import Word2VecKeyedVectors
from collections import OrderedDict


def _normalize_id(node_id):
    """Normalize any ID to string for consistent lookup"""
    return str(node_id) if node_id is not None else None
from collections import OrderedDict

def _norm_id(node_id):
    """Normalize any ID to string for consistent lookup"""
    if node_id is None:
        return None
    return str(node_id)


def graphs_embedding_from_nodes(nodes, edge_type):
    """
    Robust extractor: returns torch.LongTensor edge_index [2, E].
    nodes: list OR dict of node dicts (each node has 'id', maybe 'edges', 'ast_children', 'children').
    edge_type: string like 'ast' (matching is case-insensitive).
    
    KEY FIXES:
    1. All IDs normalized to strings
    2. Case-insensitive edge_type matching
    3. Proper neighbor list extraction from both formats
    """
    # Normalize nodes into ordered id->node map
    if isinstance(nodes, list):
        nodes_map = OrderedDict()
        for i, n in enumerate(nodes):
            nid = n.get('id', i) if isinstance(n, dict) else getattr(n, 'id', i)
            nodes_map[_norm_id(nid)] = n
    elif isinstance(nodes, dict):
        nodes_map = OrderedDict((_norm_id(k), v) for k, v in nodes.items())
    else:
        try:
            nodes_map = OrderedDict((_norm_id(n.get('id', i)), n) for i, n in enumerate(nodes))
        except Exception:
            nodes_map = OrderedDict()

    # Build id -> index mapping
    id_to_idx = {nid: idx for idx, nid in enumerate(nodes_map.keys())}

    found_labels = set()
    edges = []
    
    # Normalize edge_type for matching
    edge_type_lower = edge_type.lower()
    
    print(f"\n=== Graph Embedding Debug ===")
    print(f"Total nodes: {len(nodes_map)}")
    print(f"Looking for edge type: '{edge_type}' (normalized: '{edge_type_lower}')")

    # 1) Look for node-level dict edges: node.get('edges') -> list of dicts {source,target,label}
    node_level_edges = 0
    for idx, (nid, node) in enumerate(nodes_map.items()):
        node_dict = node if isinstance(node, dict) else None
        if node_dict:
            # Check for explicit edge dicts
            for e in node_dict.get('edges', []) or []:
                lbl = (e.get('label') or '').lower()
                found_labels.add(lbl)
                
                # CRITICAL: Case-insensitive matching
                if edge_type_lower in lbl or lbl in edge_type_lower:
                    src = _norm_id(e.get('source'))
                    tgt = _norm_id(e.get('target'))
                    if src in id_to_idx and tgt in id_to_idx:
                        edges.append((id_to_idx[src], id_to_idx[tgt]))
                        node_level_edges += 1

    print(f"Edges from node['edges']: {node_level_edges}")

    # 2) Look for neighbor lists (ast_children, children)
    neighbor_edges = {'ast_children': 0, 'children': 0}
    
    for idx, (nid, node) in enumerate(nodes_map.items()):
        # Map fields to expected labels
        field_mappings = [
            ('ast_children', 'ast'),
            ('children', 'other'),
            ('ast', 'ast'),  # Sometimes stored directly as 'ast'
            ('cfg', 'cfg'),
            ('cdg', 'cdg'),
            ('ddg', 'ddg')
        ]
        
        for field, expected_lbl in field_mappings:
            nbrs = None
            try:
                if isinstance(node, dict):
                    nbrs = node.get(field)
                else:
                    nbrs = getattr(node, field, None)
            except Exception:
                nbrs = None

            if nbrs:
                found_labels.add(expected_lbl)
                
                # CRITICAL: Check if this edge type matches what we're looking for
                # 'ast' in edge_type or edge_type in 'ast' (case-insensitive)
                expected_lbl_lower = expected_lbl.lower()
                if edge_type_lower in expected_lbl_lower or expected_lbl_lower in edge_type_lower:
                    for nb in nbrs:
                        # Handle both plain IDs and dict neighbors
                        if isinstance(nb, dict):
                            nb_id = _norm_id(nb.get('id'))
                        else:
                            nb_id = _norm_id(nb)
                        
                        if nb_id in id_to_idx:
                            edges.append((id_to_idx[nid], id_to_idx[nb_id]))
                            neighbor_edges[field] = neighbor_edges.get(field, 0) + 1

    print(f"Edges from ast_children: {neighbor_edges.get('ast_children', 0)}")
    print(f"Edges from children: {neighbor_edges.get('children', 0)}")
    print(f"Edge labels found in graph: {', '.join(sorted(found_labels)) if found_labels else '(none)'}")
    print(f"Total matched edges: {len(edges)} for requested type '{edge_type}'")
    print("=" * 50)

    if not edges:
        print("⚠️ WARNING: No edges found! Check:")
        print("  1. Are edges attached to nodes?")
        print("  2. Is edge_type matching correct?")
        print("  3. Sample node structure:")
        if nodes_map:
            sample_node = list(nodes_map.values())[0]
            if isinstance(sample_node, dict):
                print(f"     Keys: {list(sample_node.keys())}")
                print(f"     ast_children: {sample_node.get('ast_children', 'NOT FOUND')}")
                print(f"     children: {sample_node.get('children', 'NOT FOUND')}")
        return torch.empty((2, 0), dtype=torch.long)

    # Remove duplicate edges and convert to tensor
    edges = list(set(edges))  # Remove duplicates
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Final edge_index shape: {edge_index.shape}")
    return edge_index


# ---------- safe accessor ----------
# def safe_get_node_field(node, key):
#     """
#     Try robustly to fetch `key` (like 'code' or 'label') from:
#       - node dict
#       - node dict['properties'] (with Properties objects that may not support dict get signature)
#       - node object attributes (get_code(), code, label)
#       - node.properties object attributes or mapping
#     Returns the value or None.
#     """
#     # 1) if node is dict: try direct
#     if isinstance(node, dict):
#         if key in node:
#             return node.get(key)
#         # try properties child
#         props = node.get('properties')
#         if props is None:
#             return None
#         # props might be dict-like or custom object
#         # Try safe dict-style get (without default) wrapped in try/except
#         try:
#             # try props.get(key) but some Properties.get only accepts (self) -> calling with one arg raises TypeError
#             val = props.get(key)
#             return val
#         except TypeError:
#             # props.get exists but signature incompatible -> try attribute access or item access
#             try:
#                 return getattr(props, key)
#             except Exception:
#                 try:
#                     return props[key]
#                 except Exception:
#                     return None
#         except Exception:
#             # fallback: try attribute or index
#             try:
#                 return getattr(props, key)
#             except Exception:
#                 try:
#                     return props[key]
#                 except Exception:
#                     return None

#     # 2) node is object (custom Node)
#     # Try common methods / attrs
#     if hasattr(node, 'get_' + key) and callable(getattr(node, 'get_' + key)):
#         try:
#             return getattr(node, 'get_' + key)()
#         except TypeError:
#             # if it expects other args, try without
#             try:
#                 return getattr(node, 'get_' + key)()
#             except Exception:
#                 pass
#         except Exception:
#             pass

#     if hasattr(node, 'get') and callable(getattr(node, 'get')):
#         # try dict-like get
#         try:
#             return node.get(key)
#         except TypeError:
#             # get exists but doesn't accept (key, default)
#             try:
#                 return node.get(key)
#             except Exception:
#                 pass
#         except Exception:
#             pass

#     # try attribute access
#     if hasattr(node, key):
#         try:
#             return getattr(node, key)
#         except Exception:
#             pass

#     # try node.properties
#     props = getattr(node, 'properties', None)
#     if props is not None:
#         try:
#             val = props.get(key)
#             return val
#         except TypeError:
#             try:
#                 return getattr(props, key)
#             except Exception:
#                 try:
#                     return props[key]
#                 except Exception:
#                     return None
#         except Exception:
#             try:
#                 return getattr(props, key)
#             except Exception:
#                 try:
#                     return props[key]
#                 except Exception:
#                     return None

#     # nothing found
#     return None


import re
import numpy as np
import torch
from collections import OrderedDict


def safe_get_node_field(node, key):
    """
    Robustly fetch `key` from node (dict or object) and ensure string output.
    Returns string or None.
    """
    value = None
    
    # 1) Try dict access
    if isinstance(node, dict):
        if key in node:
            value = node.get(key)
        else:
            # Try properties sub-dict
            props = node.get('properties')
            if props:
                try:
                    value = props.get(key) if hasattr(props, 'get') else getattr(props, key, None)
                except (TypeError, AttributeError):
                    try:
                        value = props[key] if hasattr(props, '__getitem__') else None
                    except (KeyError, TypeError):
                        pass
    
    # 2) Try object attribute access
    else:
        # Try get_<key> method
        method_name = f'get_{key}'
        if hasattr(node, method_name):
            try:
                method = getattr(node, method_name)
                if callable(method):
                    value = method()
            except Exception:
                pass
        
        # Try direct attribute
        if value is None and hasattr(node, key):
            try:
                value = getattr(node, key)
            except Exception:
                pass
        
        # Try node.properties
        if value is None and hasattr(node, 'properties'):
            try:
                props = getattr(node, 'properties')
                value = props.get(key) if hasattr(props, 'get') else getattr(props, key, None)
            except Exception:
                try:
                    value = props[key] if hasattr(props, '__getitem__') else None
                except Exception:
                    pass
    
    # 3) CRITICAL: Ensure we return a string or None
    if value is None:
        return None
    
    # Convert to string, handling special cases
    if isinstance(value, str):
        return value
    elif isinstance(value, (int, float, bool)):
        return str(value)
    elif isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    else:
        # For complex objects, try str() but be safe
        try:
            str_value = str(value)
            # Avoid useless representations like "<Properties object at 0x...>"
            if '<' in str_value and 'object at' in str_value:
                return None
            return str_value
        except Exception:
            return None


class NodesEmbedding:
    def __init__(self, nodes_dim, keyed_vectors):
        self.nodes_dim = nodes_dim
        self.kv = keyed_vectors
        self._tok_re = re.compile(r'\w+')

    def __call__(self, nodes):
        return self.embed_nodes(nodes)

    def embed_nodes(self, nodes):
        """
        Create node embeddings from nodes (list or dict).
        Returns torch tensor of shape [N, nodes_dim].
        """
        # Normalize nodes to ordered mapping id -> node
        if isinstance(nodes, list):
            nodes_map = OrderedDict()
            for i, n in enumerate(nodes):
                nid = (n.get('id', i) if isinstance(n, dict) else getattr(n, 'id', i))
                nodes_map[nid] = n
        elif isinstance(nodes, dict):
            nodes_map = OrderedDict(nodes)
        else:
            try:
                nodes_map = OrderedDict(nodes)
            except Exception:
                nodes_map = OrderedDict()

        vectors = []
        empty_count = 0
        
        for nid, node in nodes_map.items():
            # Try several fields for node text (in priority order)
            code_text = None
            for field in ('code', 'label', 'name', 'type', 'value'):
                code_text = safe_get_node_field(node, field)
                if code_text and isinstance(code_text, str) and len(code_text.strip()) > 0:
                    break
            
            # CRITICAL: Ensure code_text is a valid string
            if not code_text or not isinstance(code_text, str):
                code_text = ""
                empty_count += 1

            # Tokenize safely
            try:
                tokens = self._tok_re.findall(code_text)
            except TypeError as e:
                # This should never happen now, but just in case
                print(f"⚠️ Tokenization error for node {nid}: {e}")
                print(f"   code_text type: {type(code_text)}, value: {repr(code_text)[:100]}")
                tokens = []

            # Vectorize tokens
            token_vecs = []
            for t in tokens:
                try:
                    if t in self.kv:
                        token_vecs.append(self.kv[t])
                except Exception:
                    try:
                        vec = self.kv.get_vector(t)
                        token_vecs.append(vec)
                    except Exception:
                        continue

            # Average or zero vector
            if token_vecs:
                mean_vec = np.mean(token_vecs, axis=0)
                # Ensure correct dimension
                if mean_vec.shape[0] != self.nodes_dim:
                    m = np.zeros(self.nodes_dim, dtype=float)
                    m[:min(self.nodes_dim, mean_vec.shape[0])] = mean_vec[:min(self.nodes_dim, mean_vec.shape[0])]
                    mean_vec = m
            else:
                mean_vec = np.zeros(self.nodes_dim, dtype=float)

            vectors.append(mean_vec)

        if empty_count > 0:
            print(f"⚠️ Warning: {empty_count}/{len(nodes_map)} nodes had empty/invalid code text")

        # Convert to tensor
        if vectors:
            arr = np.vstack(vectors).astype(np.float32)
        else:
            arr = np.zeros((0, self.nodes_dim), dtype=np.float32)

        return torch.from_numpy(arr)


# ---------- resilient NodesEmbedding ----------
class NodesEmbedding:
    def __init__(self, nodes_dim, keyed_vectors):
        self.nodes_dim = nodes_dim
        self.kv = keyed_vectors
        self._tok_re = re.compile(r'\w+')

    def __call__(self, nodes):
        return self.embed_nodes(nodes)

    def embed_nodes(self, nodes):
        # Normalize nodes to ordered mapping id -> node
        if isinstance(nodes, list):
            nodes_map = OrderedDict()
            for i, n in enumerate(nodes):
                nid = (n.get('id', i) if isinstance(n, dict) else getattr(n, 'id', i))
                nodes_map[nid] = n
        elif isinstance(nodes, dict):
            nodes_map = OrderedDict(nodes)
        else:
            try:
                nodes_map = OrderedDict(nodes)
            except Exception:
                nodes_map = OrderedDict()

        vectors = []
        for nid, node in nodes_map.items():
            # Try several fields for node text
            code_text = None
            for field in ('code', 'label', 'name', 'value'):
                code_text = safe_get_node_field(node, field)
                if code_text:
                    break
            if not code_text:
                code_text = ''

            # Tokenize and vectorize
            tokens = self._tok_re.findall(code_text or "")
            token_vecs = []
            for t in tokens:
                try:
                    if t in self.kv:
                        token_vecs.append(self.kv[t])
                except Exception:
                    try:
                        vec = self.kv.get_vector(t)
                        token_vecs.append(vec)
                    except Exception:
                        continue

            if token_vecs:
                mean_vec = np.mean(token_vecs, axis=0)
                if mean_vec.shape[0] != self.nodes_dim:
                    m = np.zeros(self.nodes_dim, dtype=float)
                    m[:min(self.nodes_dim, mean_vec.shape[0])] = mean_vec[:min(self.nodes_dim, mean_vec.shape[0])]
                    mean_vec = m
            else:
                mean_vec = np.zeros(self.nodes_dim, dtype=float)

            vectors.append(mean_vec)

        if vectors:
            arr = np.vstack(vectors).astype(np.float32)
        else:
            arr = np.zeros((0, self.nodes_dim), dtype=np.float32)

        return torch.from_numpy(arr)






# Wrapper function for backward compatibility
def GraphsEmbedding(edge_type):
    """Wrapper that creates a callable using graphs_embedding_from_nodes"""
    def wrapper(nodes):
        return graphs_embedding_from_nodes(nodes, edge_type)
    return wrapper





# import torch
# from torch_geometric.data import Data

# def build_edge_index_from_nodes(nodes, edge_field_candidates=('ast_children','children','children_ids','edges','succs','succ')):
#     """
#     Accepts 'nodes' as a dict (id -> node dict) OR list of node dicts.
#     Returns torch.LongTensor of shape [2, E] or empty tensor.
#     """
#     # Normalize to dict id -> node
#     if isinstance(nodes, list):
#         id_map = {n.get('id', i): n for i, n in enumerate(nodes)}
#     else:
#         id_map = dict(nodes)

#     id_to_idx = {}
#     for i, nid in enumerate(id_map.keys()):
#         id_to_idx[nid] = i

#     edges = []
#     for i, (nid, n) in enumerate(id_map.items()):
#         for field in edge_field_candidates:
#             neighs = n.get(field)
#             if not neighs:
#                 continue
#             for nb in neighs:
#                 if isinstance(nb, dict):
#                     nb_id = nb.get('id')
#                 else:
#                     nb_id = nb
#                 if nb_id in id_to_idx:
#                     edges.append((i, id_to_idx[nb_id]))

#     if not edges:
#         return torch.empty((2, 0), dtype=torch.long)
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#     return edge_index

# def nodes_to_input(nodes, target, nodes_dim, keyed_vectors, edge_type):
#     """
#     Robust nodes -> PyG Data builder.
#     Accepts nodes as list OR dict. Uses GraphsEmbedding first (which expects dict)
#     and falls back to build_edge_index_from_nodes when needed.
#     """
#     # Normalize nodes to dict for embedder/graph builder
#     if isinstance(nodes, list):
#         nodes_dict = {n.get('id', i): n for i, n in enumerate(nodes)}
#     else:
#         nodes_dict = dict(nodes)

#     # Create embeddings/builders as before
#     nodes_embedding = NodesEmbedding(nodes_dim, keyed_vectors)
#     graphs_embedding = GraphsEmbedding(edge_type)
#     label = torch.tensor([target]).float()

#     # NodesEmbedding (your existing implementation expects dict and iterates items())
#     try:
#         x = nodes_embedding(nodes_dict)  # shape [N, nodes_dim]
#     except Exception as e:
#         print("NodesEmbedding error:", e)
#         # Try to coerce to list and fallback to zeros
#         if isinstance(nodes, list):
#             N = len(nodes)
#         else:
#             N = len(nodes_dict)
#         x = torch.zeros((N, nodes_dim), dtype=torch.float)

#     # Graph building
#     edge_index = None
#     try:
#         edge_index = graphs_embedding(nodes_dict)
#     except Exception as e:
#         print("GraphsEmbedding raised:", e)
#         edge_index = None

#     # Fallback if no edges produced
#     if edge_index is None or (hasattr(edge_index, 'numel') and edge_index.numel() == 0) or (isinstance(edge_index, torch.Tensor) and edge_index.shape[1] == 0):
#         print("GraphsEmbedding produced no edges — using fallback builder.")
#         edge_index = build_edge_index_from_nodes(nodes_dict)
#         if edge_index.numel() == 0:
#             # Last resort: create self-loops so models expecting edges won't crash
#             n = x.size(0) if x is not None else (len(nodes_dict))
#             idx = torch.arange(n, dtype=torch.long)
#             edge_index = torch.stack([idx, idx], dim=0)

#     return Data(x=x, edge_index=edge_index, y=label)




import torch
from torch_geometric.data import Data


def _norm_id(node_id):
    """Normalize any ID to string"""
    return str(node_id) if node_id is not None else None


def build_edge_index_from_nodes(nodes, edge_field_candidates=('ast_children', 'children', 'ast', 'edges', 'cfg', 'cdg', 'ddg')):
    """
    Fallback edge builder from node neighbor lists.
    Accepts 'nodes' as dict (id -> node) OR list of node dicts.
    Returns torch.LongTensor [2, E] or empty tensor.
    """
    # Normalize to dict id -> node
    if isinstance(nodes, list):
        id_map = {_norm_id(n.get('id', i)): n for i, n in enumerate(nodes)}
    else:
        id_map = {_norm_id(k): v for k, v in nodes.items()}

    id_to_idx = {nid: i for i, nid in enumerate(id_map.keys())}
    edges = []
    
    for i, (nid, n) in enumerate(id_map.items()):
        for field in edge_field_candidates:
            neighs = n.get(field) if isinstance(n, dict) else getattr(n, field, None)
            if not neighs:
                continue
            
            for nb in neighs:
                # Handle dict neighbors or plain IDs
                if isinstance(nb, dict):
                    nb_id = _norm_id(nb.get('id'))
                else:
                    nb_id = _norm_id(nb)
                
                if nb_id in id_to_idx:
                    edges.append((i, id_to_idx[nb_id]))

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Remove duplicates
    edges = list(set(edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def GraphsEmbedding(edge_type):
    """Wrapper for graphs_embedding_from_nodes"""
    def wrapper(nodes):
        return graphs_embedding_from_nodes(nodes, edge_type)
    return wrapper


def nodes_to_input(nodes, target, nodes_dim, keyed_vectors, edge_type):
    """
    Robust nodes -> PyG Data builder.
    Accepts nodes as list OR dict.
    
    Process:
    1. Normalize nodes to dict
    2. Create node embeddings (X)
    3. Create edge_index using GraphsEmbedding
    4. Fallback to build_edge_index_from_nodes if needed
    5. Last resort: self-loops
    """
    # Normalize nodes to dict
    if isinstance(nodes, list):
        nodes_dict = {_norm_id(n.get('id', i)): n for i, n in enumerate(nodes)}
    else:
        nodes_dict = {_norm_id(k): v for k, v in nodes.items()}

    print(f"\n=== Creating Input for Sample ===")
    print(f"Target: {target}")
    print(f"Nodes: {len(nodes_dict)}")
    print(f"Edge type requested: '{edge_type}'")

    # Create embedders
    nodes_embedding = NodesEmbedding(nodes_dim, keyed_vectors)
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    # 1. Create node features
    try:
        x = nodes_embedding(nodes_dict)  # [N, nodes_dim]
        print(f"Node features shape: {x.shape}")
    except Exception as e:
        print(f"❌ NodesEmbedding error: {e}")
        N = len(nodes_dict)
        x = torch.zeros((N, nodes_dim), dtype=torch.float)
        print(f"Using zero embeddings: {x.shape}")

    # 2. Create edge_index (primary method)
    edge_index = None
    try:
        edge_index = graphs_embedding(nodes_dict)
        if edge_index is not None and edge_index.numel() > 0:
            print(f"✓ GraphsEmbedding produced edges: {edge_index.shape}")
    except Exception as e:
        print(f"❌ GraphsEmbedding error: {e}")

    # 3. Fallback to neighbor list extraction
    if edge_index is None or edge_index.numel() == 0:
        print("⚠️ GraphsEmbedding produced no edges, trying fallback...")
        edge_index = build_edge_index_from_nodes(nodes_dict)
        if edge_index.numel() > 0:
            print(f"✓ Fallback produced edges: {edge_index.shape}")

    # 4. Last resort: self-loops
    if edge_index is None or edge_index.numel() == 0:
        print("⚠️ No edges from any method, creating self-loops...")
        n = x.size(0)
        idx = torch.arange(n, dtype=torch.long)
        edge_index = torch.stack([idx, idx], dim=0)
        print(f"✓ Self-loops created: {edge_index.shape}")

    print(f"Final graph: x={x.shape}, edge_index={edge_index.shape}, y={label.item()}")
    print("=" * 50 + "\n")

    return Data(x=x, edge_index=edge_index, y=label)