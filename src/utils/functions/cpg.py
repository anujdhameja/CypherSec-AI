

from collections import OrderedDict
from ..objects.cpg.function import Function
from ..objects.cpg.node import Node



def order_nodes(nodes, max_nodes):
    # Handle empty nodes case
    if not nodes:
        return {}

    # Sort nodes by column number (treat None as -1)
    nodes_by_column = sorted(
        nodes.items(),
        key=lambda n: n[1].get_column_number() if n[1].get_column_number() is not None else -1
    )

    # Then sort by line number (treat None as -1)
    nodes_by_line = sorted(
        nodes_by_column,
        key=lambda n: n[1].get_line_number() if n[1].get_line_number() is not None else -1
    )

    # Set order property and create ordered dictionary
    ordered_nodes = {}
    for i, (node_id, node) in enumerate(nodes_by_line):
        if not isinstance(node, str):  # Ensure node is not a string
            node.order = i
            ordered_nodes[node_id] = node

    # Return only up to max_nodes
    return dict(list(ordered_nodes.items())[:max_nodes]) if max_nodes else ordered_nodes






def filter_nodes(nodes):
    if not nodes:
        print("⚠️ No nodes to filter")
        return {}

    filtered = {}
    for node_id, node in nodes.items():
        if not isinstance(node, Node):
            print(f"⚠️ Skipping non-Node: {node}")
            continue
        
        # Add more debug prints here to check node properties
        # print(f"Node {node_id}: label='{getattr(node, 'label', 'N/A')}', type={type(node)}")
        
        # Keep the node for now (remove any filtering temporarily)
        filtered[node_id] = node

    # print(f"Kept {len(filtered)}/{len(nodes)} nodes after filtering")
    return filtered










# --- Helpers ---------------------------------------------------------
def _node_to_dict(node, fallback_key):
    """
    Convert node object (dict or custom Node instance) into a plain dict.
    Ensure an 'id' key exists.
    """
    # If it's already a dict, shallow-copy to avoid mutating original
    if isinstance(node, dict):
        d = dict(node)
    else:
        d = {}
        # Try common attribute names
        for attr in ("id", "node_id", "nid", "label", "code", "name"):
            if hasattr(node, attr):
                try:
                    d[attr] = getattr(node, attr)
                except Exception:
                    pass
        # If object has a to_dict / asdict style method, try it
        if hasattr(node, "to_dict") and callable(getattr(node, "to_dict")):
            try:
                for k, v in node.to_dict().items():
                    if k not in d:
                        d[k] = v
            except Exception:
                pass
        # finally, try __dict__ fallback
        if hasattr(node, "__dict__"):
            try:
                for k, v in node.__dict__.items():
                    if k not in d:
                        d[k] = v
            except Exception:
                pass

    # Ensure id exists
    if 'id' not in d or d.get('id') is None:
        d['id'] = fallback_key
    return d

def _attach_neighbor(holders, src_id, tgt_id, label):
    """Attach tgt_id to holders[src_id] under ast_children or children depending on label"""
    if src_id not in holders or tgt_id not in holders:
        return
    if label and 'ast' in label.lower():
        holders[src_id].setdefault('ast_children', []).append(tgt_id)
    else:
        holders[src_id].setdefault('children', []).append(tgt_id)




# def parse_to_nodes(cpg, max_nodes=500):
#     """
#     Parse CPG and return an ordered LIST of plain node dicts with:
#       - 'id', 'label', 'code', ...
#       - 'children' (non-AST edges)
#       - 'ast_children' (AST edges)
#     Implementation notes:
#       - We keep the original Node-like objects for ordering (order_nodes expects objects
#         with methods like get_column_number()), so we only convert to dicts after ordering.
#       - Edges are collected in neighbors_map and attached after ordering.
#     """
#     print("CPG keys:", list(cpg.keys()))
#     all_nodes_orig = {}   # id -> original node object (whatever Function.get_nodes returns)
#     neighbors_map = {}    # id -> {'children': [...], 'ast_children': [...]}

#     def add_node_orig(nid, node_obj):
#         # store original node object for ordering
#         if nid not in all_nodes_orig:
#             all_nodes_orig[nid] = node_obj
#         # ensure neighbors_map entry exists
#         if nid not in neighbors_map:
#             neighbors_map[nid] = {'children': [], 'ast_children': []}

#     def record_edge(src, tgt, label):
#         if src is None or tgt is None:
#             return
#         # ensure entries
#         if src not in neighbors_map:
#             neighbors_map[src] = {'children': [], 'ast_children': []}
#         if 'ast' in (label or '').lower():
#             neighbors_map[src]['ast_children'].append(tgt)
#         else:
#             neighbors_map[src]['children'].append(tgt)

#     # --- collect nodes & edges (preserve original node objects) ---
#     if "functions" in cpg:
#         print("Found 'functions' in CPG")
#         for function in cpg["functions"]:
#             raw_nodes = Function(function).get_nodes()
#             filtered = filter_nodes(raw_nodes)

#             # filtered might be dict or list
#             if isinstance(filtered, dict):
#                 for k, node_obj in filtered.items():
#                     # attempt to get id from node_obj if possible, otherwise use key
#                     nid = None
#                     if isinstance(node_obj, dict):
#                         nid = node_obj.get('id', k)
#                     else:
#                         # Node object: try common attributes
#                         nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or k
#                     add_node_orig(nid, node_obj)
#             else:
#                 for i, node_obj in enumerate(filtered):
#                     nid = None
#                     if isinstance(node_obj, dict):
#                         nid = node_obj.get('id', i)
#                     else:
#                         nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or i
#                     add_node_orig(nid, node_obj)

#             # attach edges declared in this function (function may be a dict-like)
#             for e in (function.get('edges', []) or []):
#                 src = e.get('source')
#                 tgt = e.get('target')
#                 lbl = e.get('label', '')
#                 record_edge(src, tgt, lbl)

#             print(f"  Function '{function.get('function', '<unknown>')}' -> nodes collected so far: {len(all_nodes_orig)}")

#     elif "nodes" in cpg:
#         print("Found top-level 'nodes' in CPG")
#         valid_nodes = [n for n in cpg["nodes"] if isinstance(n, (dict, object))]
#         func = Function({"nodes": valid_nodes, "edges": cpg.get("edges", [])})
#         filtered = filter_nodes(func.get_nodes())

#         if isinstance(filtered, dict):
#             for k, node_obj in filtered.items():
#                 nid = None
#                 if isinstance(node_obj, dict):
#                     nid = node_obj.get('id', k)
#                 else:
#                     nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or k
#                 add_node_orig(nid, node_obj)
#         else:
#             for i, node_obj in enumerate(filtered):
#                 nid = None
#                 if isinstance(node_obj, dict):
#                     nid = node_obj.get('id', i)
#                 else:
#                     nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or i
#                 add_node_orig(nid, node_obj)

#         # top-level edges
#         for e in (cpg.get('edges', []) or []):
#             src = e.get('source')
#             tgt = e.get('target')
#             lbl = e.get('label', '')
#             record_edge(src, tgt, lbl)

#         print(f"Kept {len(all_nodes_orig)} nodes after filtering")

#     else:
#         print("⚠️ Unknown CPG format — no 'functions' or 'nodes' key")
#         return []

#     # --- Order nodes using existing helper (order_nodes expects original objects) ---
#     ordered = order_nodes(all_nodes_orig, max_nodes)
#     print(f"Ordered {len(ordered)} nodes (ordered is dict id->node_obj)")

#     # --- Convert ordered Node objects to plain dicts and attach neighbor lists ---
#     ordered_list = []
#     for nid, node_obj in ordered.items():
#         # convert node_obj -> dict with safe approach (try dict first, then attributes)
#         if isinstance(node_obj, dict):
#             node_dict = dict(node_obj)  # shallow copy
#         else:
#             node_dict = {}
#             # pick common attributes if available
#             for attr in ("id", "label", "code", "name", "type"):
#                 if hasattr(node_obj, attr):
#                     try:
#                         node_dict[attr] = getattr(node_obj, attr)
#                     except Exception:
#                         pass
#             # try to_dict or __dict__
#             if hasattr(node_obj, "to_dict") and callable(getattr(node_obj, "to_dict")):
#                 try:
#                     for k, v in node_obj.to_dict().items():
#                         if k not in node_dict:
#                             node_dict[k] = v
#                 except Exception:
#                     pass
#             elif hasattr(node_obj, "__dict__"):
#                 try:
#                     for k, v in node_obj.__dict__.items():
#                         if k not in node_dict:
#                             node_dict[k] = v
#                 except Exception:
#                     pass

#         # ensure id present and neighbor lists
#         node_dict['id'] = node_dict.get('id', nid)
#         node_dict.setdefault('children', [])
#         node_dict.setdefault('ast_children', [])

#         # attach neighbors discovered earlier if present
#         neigh = neighbors_map.get(node_dict['id'], {})
#         if neigh:
#             if neigh.get('children'):
#                 node_dict['children'].extend(neigh.get('children'))
#             if neigh.get('ast_children'):
#                 node_dict['ast_children'].extend(neigh.get('ast_children'))

#         ordered_list.append(node_dict)

#     print(f"Returning {len(ordered_list)} ordered node dicts")
#     return ordered_list



def _normalize_id(node_id):
    """Normalize IDs to strings for consistent lookup"""
    if node_id is None:
        return None
    return str(node_id)


def parse_to_nodes(cpg, max_nodes=500):
    """
    Parse CPG and return an ordered LIST of plain node dicts with:
      - 'id', 'label', 'code', ...
      - 'children' (non-AST edges)
      - 'ast_children' (AST edges)
    
    KEY FIX: All IDs are normalized to strings for consistent lookup
    CRITICAL FIX: Preserve original 'code' field from raw nodes
    """
    print("CPG keys:", list(cpg.keys()))
    all_nodes_orig = {}   # normalized_id -> original node object
    neighbors_map = {}    # normalized_id -> {'children': [...], 'ast_children': [...]}
    raw_node_data = {}    # normalized_id -> original raw node dict (preserves 'code')

    def add_node_orig(nid, node_obj):
        """Store node with normalized ID"""
        normalized_id = _normalize_id(nid)
        if normalized_id not in all_nodes_orig:
            all_nodes_orig[normalized_id] = node_obj
        if normalized_id not in neighbors_map:
            neighbors_map[normalized_id] = {'children': [], 'ast_children': []}

    def record_edge(src, tgt, label):
        """Record edge with normalized IDs"""
        if src is None or tgt is None:
            return
        
        # CRITICAL: Normalize both source and target IDs
        norm_src = _normalize_id(src)
        norm_tgt = _normalize_id(tgt)
        
        # Ensure entries exist
        if norm_src not in neighbors_map:
            neighbors_map[norm_src] = {'children': [], 'ast_children': []}
        
        # Normalize label matching (case-insensitive)
        label_lower = (label or '').lower()
        if 'ast' in label_lower:
            neighbors_map[norm_src]['ast_children'].append(norm_tgt)
        else:
            neighbors_map[norm_src]['children'].append(norm_tgt)

    # --- Collect nodes & edges ---
    if "functions" in cpg:
        print("Found 'functions' in CPG")
        for function in cpg["functions"]:
            raw_nodes = Function(function).get_nodes()
            filtered = filter_nodes(raw_nodes)

            # Add nodes with normalized IDs
            if isinstance(filtered, dict):
                for k, node_obj in filtered.items():
                    nid = None
                    if isinstance(node_obj, dict):
                        nid = node_obj.get('id', k)
                    else:
                        nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or k
                    add_node_orig(nid, node_obj)
            else:
                for i, node_obj in enumerate(filtered):
                    nid = None
                    if isinstance(node_obj, dict):
                        nid = node_obj.get('id', i)
                    else:
                        nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or i
                    add_node_orig(nid, node_obj)

            # CRITICAL: Record edges from function level
            for e in (function.get('edges', []) or []):
                src = e.get('source')
                tgt = e.get('target')
                lbl = e.get('label', '')
                record_edge(src, tgt, lbl)

            print(f"  Function '{function.get('function', '<unknown>')}' -> nodes: {len(all_nodes_orig)}, edges recorded: {sum(len(v['children']) + len(v['ast_children']) for v in neighbors_map.values())}")

    elif "nodes" in cpg:
        print("Found top-level 'nodes' in CPG")
        valid_nodes = [n for n in cpg["nodes"] if isinstance(n, (dict, object))]
        
        # CRITICAL FIX: Store original raw node data before processing
        for raw_node in valid_nodes:
            if isinstance(raw_node, dict) and 'id' in raw_node:
                raw_id = _normalize_id(raw_node['id'])
                raw_node_data[raw_id] = raw_node  # Preserve original dict with 'code'
        
        func = Function({"nodes": valid_nodes, "edges": cpg.get("edges", [])})
        filtered = filter_nodes(func.get_nodes())

        if isinstance(filtered, dict):
            for k, node_obj in filtered.items():
                nid = None
                if isinstance(node_obj, dict):
                    nid = node_obj.get('id', k)
                else:
                    nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or k
                add_node_orig(nid, node_obj)
        else:
            for i, node_obj in enumerate(filtered):
                nid = None
                if isinstance(node_obj, dict):
                    nid = node_obj.get('id', i)
                else:
                    nid = getattr(node_obj, 'id', None) or getattr(node_obj, 'node_id', None) or i
                add_node_orig(nid, node_obj)

        # Top-level edges
        for e in (cpg.get('edges', []) or []):
            src = e.get('source')
            tgt = e.get('target')
            lbl = e.get('label', '')
            record_edge(src, tgt, lbl)

        print(f"Kept {len(all_nodes_orig)} nodes, edges recorded: {sum(len(v['children']) + len(v['ast_children']) for v in neighbors_map.values())}")

    else:
        print("⚠️ Unknown CPG format — no 'functions' or 'nodes' key")
        return []

    # --- Order nodes ---
    ordered = order_nodes(all_nodes_orig, max_nodes)
    print(f"Ordered {len(ordered)} nodes")

    # --- Convert to plain dicts and attach neighbors ---
    ordered_list = []
    edge_attachment_count = 0
    
    for nid, node_obj in ordered.items():
        # Convert node_obj to dict
        if isinstance(node_obj, dict):
            node_dict = dict(node_obj)
        else:
            node_dict = {}
            for attr in ("id", "label", "code", "name", "type"):
                if hasattr(node_obj, attr):
                    try:
                        node_dict[attr] = getattr(node_obj, attr)
                    except Exception:
                        pass
            if hasattr(node_obj, "to_dict") and callable(getattr(node_obj, "to_dict")):
                try:
                    for k, v in node_obj.to_dict().items():
                        if k not in node_dict:
                            node_dict[k] = v
                except Exception:
                    pass
            elif hasattr(node_obj, "__dict__"):
                try:
                    for k, v in node_obj.__dict__.items():
                        if k not in node_dict:
                            node_dict[k] = v
                except Exception:
                    pass

        # Ensure ID is normalized and neighbors are initialized
        normalized_id = _normalize_id(node_dict.get('id', nid))
        node_dict['id'] = normalized_id
        node_dict.setdefault('children', [])
        node_dict.setdefault('ast_children', [])

        # CRITICAL FIX: Restore original 'code' field from raw node data
        if normalized_id in raw_node_data:
            original_raw = raw_node_data[normalized_id]
            if 'code' in original_raw:
                node_dict['code'] = original_raw['code']  # Restore the code field!

        # CRITICAL: Attach neighbors using normalized ID
        neigh = neighbors_map.get(normalized_id, {})
        if neigh:
            if neigh.get('children'):
                node_dict['children'].extend(neigh.get('children'))
                edge_attachment_count += len(neigh.get('children'))
            if neigh.get('ast_children'):
                node_dict['ast_children'].extend(neigh.get('ast_children'))
                edge_attachment_count += len(neigh.get('ast_children'))

        ordered_list.append(node_dict)

    print(f"Returning {len(ordered_list)} ordered node dicts with {edge_attachment_count} edges attached")
    return ordered_list