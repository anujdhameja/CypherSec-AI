"""
Final Checker - Comprehensive CPG Pipeline Validation
Validates data at each stage: Raw CPG → Parsed Nodes → Graph Input
"""

import os
import json
import pickle
import pandas as pd
import torch
from pathlib import Path
from collections import Counter, defaultdict
import networkx as nx
from torch_geometric.utils import to_networkx


class CPGPipelineChecker:
    def __init__(self, paths):
        """
        paths should contain:
        - cpg: raw CPG pickle files
        - input: processed input files
        """
        self.paths = paths
        self.results = {
            'raw_cpg': {},
            'processed_input': {},
            'validation': {}
        }
    
    def check_raw_cpg(self, cpg_file):
        """Check structure of raw CPG file"""
        print(f"\n{'='*80}")
        print(f"CHECKING RAW CPG: {cpg_file}")
        print(f"{'='*80}")
        
        file_path = os.path.join(self.paths['cpg'], cpg_file)
        df = pd.read_pickle(file_path)
        
        print(f"\n1. DATAFRAME STRUCTURE")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Rows: {len(df)}")
        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
        
        # Check CPG structure
        print(f"\n2. CPG STRUCTURE (First Sample)")
        sample_cpg = df['cpg'].iloc[0]
        
        if isinstance(sample_cpg, dict):
            print(f"   CPG keys: {list(sample_cpg.keys())}")
            
            # Check edges format
            if 'edges' in sample_cpg:
                print(f"\n   EDGES (Top-level):")
                print(f"   - Total edges: {len(sample_cpg['edges'])}")
                
                if sample_cpg['edges']:
                    edge_sample = sample_cpg['edges'][0]
                    print(f"   - Edge format: {type(edge_sample)}")
                    if isinstance(edge_sample, dict):
                        print(f"   - Edge keys: {list(edge_sample.keys())}")
                        print(f"   - Sample edge: {edge_sample}")
                    
                    # Analyze edge labels
                    edge_labels = [e.get('label', 'unknown') for e in sample_cpg['edges'][:100]]
                    label_counts = Counter(edge_labels)
                    print(f"   - Edge label distribution: {dict(label_counts)}")
            
            # Check nodes format
            if 'nodes' in sample_cpg:
                print(f"\n   NODES (Top-level):")
                print(f"   - Total nodes: {len(sample_cpg['nodes'])}")
                
                if sample_cpg['nodes']:
                    node_sample = sample_cpg['nodes'][0]
                    print(f"   - Node format: {type(node_sample)}")
                    if isinstance(node_sample, dict):
                        print(f"   - Node keys: {list(node_sample.keys())}")
                        
                        # Check if nodes have embedded edges
                        has_ast_children = 'ast_children' in node_sample
                        has_children = 'children' in node_sample
                        has_edges = 'edges' in node_sample
                        
                        print(f"   - Has 'ast_children': {has_ast_children}")
                        print(f"   - Has 'children': {has_children}")
                        print(f"   - Has 'edges': {has_edges}")
                        
                        if has_ast_children:
                            print(f"   - Sample ast_children: {node_sample.get('ast_children', [])[:5]}")
                        if has_children:
                            print(f"   - Sample children: {node_sample.get('children', [])[:5]}")
                        if has_edges:
                            print(f"   - Sample edges: {node_sample.get('edges', [])[:2]}")
            
            # Check if it's function-based structure
            if 'functions' in sample_cpg:
                print(f"\n   FUNCTIONS:")
                print(f"   - Total functions: {len(sample_cpg['functions'])}")
        
        return {
            'file': cpg_file,
            'rows': len(df),
            'cpg_format': 'dict' if isinstance(sample_cpg, dict) else type(sample_cpg),
            'has_edges': 'edges' in sample_cpg if isinstance(sample_cpg, dict) else False,
            'has_nodes': 'nodes' in sample_cpg if isinstance(sample_cpg, dict) else False
        }
    
    def check_processed_input(self, input_file):
        """Check structure of processed input file"""
        print(f"\n{'='*80}")
        print(f"CHECKING PROCESSED INPUT: {input_file}")
        print(f"{'='*80}")
        
        file_path = os.path.join(self.paths['input'], input_file)
        df = pd.read_pickle(file_path)
        
        print(f"\n1. DATAFRAME STRUCTURE")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Rows: {len(df)}")
        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
        
        # Analyze input graphs
        print(f"\n2. GRAPH STRUCTURE (First 5 Samples)")
        
        graph_stats = {
            'num_nodes': [],
            'num_edges': [],
            'node_features_dim': [],
            'has_self_loops': [],
            'is_connected': []
        }
        
        for idx in range(min(5, len(df))):
            sample = df.iloc[idx]
            graph = sample['input']
            target = sample['target']
            
            num_nodes = graph.x.shape[0]
            num_edges = graph.edge_index.shape[1]
            feature_dim = graph.x.shape[1]
            
            # Check for self-loops
            edge_index = graph.edge_index
            self_loops = (edge_index[0] == edge_index[1]).sum().item()
            
            # Check connectivity
            G = to_networkx(graph, to_undirected=True)
            is_connected = nx.is_connected(G)
            num_components = nx.number_connected_components(G)
            
            print(f"\n   Sample {idx} (target={target}):")
            print(f"   - Nodes: {num_nodes}")
            print(f"   - Edges: {num_edges}")
            print(f"   - Features dim: {feature_dim}")
            print(f"   - Self-loops: {self_loops}")
            print(f"   - Connected: {is_connected} ({num_components} components)")
            print(f"   - Edge ratio: {num_edges}/{num_nodes} = {num_edges/num_nodes:.2f}")
            
            # Show sample edges
            if num_edges > 0:
                print(f"   - First 5 edges: {edge_index[:, :5].tolist()}")
            
            graph_stats['num_nodes'].append(num_nodes)
            graph_stats['num_edges'].append(num_edges)
            graph_stats['node_features_dim'].append(feature_dim)
            graph_stats['has_self_loops'].append(self_loops > 0)
            graph_stats['is_connected'].append(is_connected)
        
        # Overall statistics
        print(f"\n3. OVERALL STATISTICS (All {len(df)} samples)")
        all_graphs = df['input'].tolist()
        
        all_nodes = [g.x.shape[0] for g in all_graphs]
        all_edges = [g.edge_index.shape[1] for g in all_graphs]
        
        print(f"   Nodes: min={min(all_nodes)}, max={max(all_nodes)}, avg={sum(all_nodes)/len(all_nodes):.1f}")
        print(f"   Edges: min={min(all_edges)}, max={max(all_edges)}, avg={sum(all_edges)/len(all_edges):.1f}")
        
        # Check for graphs with no edges
        no_edges = sum(1 for e in all_edges if e == 0)
        if no_edges > 0:
            print(f"   ⚠️ WARNING: {no_edges} graphs have NO edges!")
        
        # Check edge/node ratio distribution
        ratios = [e/n if n > 0 else 0 for e, n in zip(all_edges, all_nodes)]
        print(f"   Edge/Node ratio: min={min(ratios):.2f}, max={max(ratios):.2f}, avg={sum(ratios)/len(ratios):.2f}")
        
        return {
            'file': input_file,
            'rows': len(df),
            'graphs_with_edges': sum(1 for e in all_edges if e > 0),
            'graphs_without_edges': no_edges,
            'avg_nodes': sum(all_nodes) / len(all_nodes),
            'avg_edges': sum(all_edges) / len(all_edges)
        }
    
    def compare_raw_vs_processed(self, cpg_file, input_file):
        """Compare raw CPG with processed input to validate transformation"""
        print(f"\n{'='*80}")
        print(f"COMPARING RAW vs PROCESSED")
        print(f"{'='*80}")
        
        # Load both
        cpg_df = pd.read_pickle(os.path.join(self.paths['cpg'], cpg_file))
        input_df = pd.read_pickle(os.path.join(self.paths['input'], input_file))
        
        print(f"\n1. ROW COUNT COMPARISON")
        print(f"   Raw CPG rows: {len(cpg_df)}")
        print(f"   Processed rows: {len(input_df)}")
        if len(cpg_df) != len(input_df):
            print(f"   ⚠️ WARNING: Row count mismatch! {len(cpg_df) - len(input_df)} rows lost")
        
        print(f"\n2. EDGE TRANSFORMATION ANALYSIS")
        
        # Sample comparison
        for idx in range(min(3, len(cpg_df), len(input_df))):
            print(f"\n   Sample {idx}:")
            
            # Raw CPG edges
            raw_cpg = cpg_df.iloc[idx]['cpg']
            if isinstance(raw_cpg, dict) and 'edges' in raw_cpg:
                raw_edge_count = len(raw_cpg['edges'])
                raw_edge_labels = Counter([e.get('label') for e in raw_cpg['edges']])
                print(f"   Raw CPG edges: {raw_edge_count}")
                print(f"   Raw edge labels: {dict(raw_edge_labels)}")
            
            # Processed graph edges
            processed_graph = input_df.iloc[idx]['input']
            processed_edge_count = processed_graph.edge_index.shape[1]
            print(f"   Processed edges: {processed_edge_count}")
            
            # Node count
            if 'nodes' in raw_cpg:
                raw_node_count = len(raw_cpg['nodes'])
                processed_node_count = processed_graph.x.shape[0]
                print(f"   Raw nodes: {raw_node_count}")
                print(f"   Processed nodes: {processed_node_count}")
    
    def validate_graph_quality(self, input_file):
        """Validate that graphs are properly formed"""
        print(f"\n{'='*80}")
        print(f"GRAPH QUALITY VALIDATION")
        print(f"{'='*80}")
        
        file_path = os.path.join(self.paths['input'], input_file)
        df = pd.read_pickle(file_path)
        
        issues = defaultdict(list)
        
        for idx, row in df.iterrows():
            graph = row['input']
            
            # Check 1: Edge indices within bounds
            if graph.edge_index.numel() > 0:
                max_edge_idx = graph.edge_index.max().item()
                num_nodes = graph.x.shape[0]
                if max_edge_idx >= num_nodes:
                    issues['invalid_edge_indices'].append(idx)
            
            # Check 2: No edges
            if graph.edge_index.shape[1] == 0:
                issues['no_edges'].append(idx)
            
            # Check 3: All self-loops
            if graph.edge_index.shape[1] > 0:
                all_self_loops = (graph.edge_index[0] == graph.edge_index[1]).all()
                if all_self_loops:
                    issues['all_self_loops'].append(idx)
            
            # Check 4: Disconnected graph
            if graph.edge_index.shape[1] > 0:
                G = to_networkx(graph, to_undirected=True)
                if not nx.is_connected(G):
                    issues['disconnected'].append(idx)
            
            # Check 5: NaN in features
            if torch.isnan(graph.x).any():
                issues['nan_features'].append(idx)
        
        print(f"\nValidation Results for {len(df)} graphs:")
        print(f"{'✓' if not issues else '✗'} Overall Status: {'PASS' if not issues else 'ISSUES FOUND'}")
        
        if issues:
            for issue_type, indices in issues.items():
                print(f"\n   ⚠️ {issue_type}: {len(indices)} graphs")
                if len(indices) <= 5:
                    print(f"      Indices: {indices}")
                else:
                    print(f"      Indices (first 5): {indices[:5]}")
        else:
            print("\n   ✓ No edge index issues")
            print("   ✓ All graphs have edges")
            print("   ✓ No all-self-loop graphs")
            print("   ✓ All graphs connected")
            print("   ✓ No NaN features")
        
        return issues
    
    def run_full_check(self, cpg_file, input_file):
        """Run complete validation pipeline"""
        print(f"\n{'#'*80}")
        print(f"# FULL PIPELINE CHECK")
        print(f"# CPG: {cpg_file}")
        print(f"# Input: {input_file}")
        print(f"{'#'*80}")
        
        # Stage 1: Check raw CPG
        raw_results = self.check_raw_cpg(cpg_file)
        
        # Stage 2: Check processed input
        processed_results = self.check_processed_input(input_file)
        
        # Stage 3: Compare
        self.compare_raw_vs_processed(cpg_file, input_file)
        
        # Stage 4: Validate quality
        issues = self.validate_graph_quality(input_file)
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"✓ Raw CPG loaded: {raw_results['rows']} samples")
        print(f"✓ Processed input loaded: {processed_results['rows']} samples")
        print(f"✓ Graphs with edges: {processed_results['graphs_with_edges']}/{processed_results['rows']}")
        
        if issues:
            print(f"⚠️ Quality issues found: {len(issues)} types")
        else:
            print(f"✓ All quality checks passed")
        
        return {
            'raw': raw_results,
            'processed': processed_results,
            'issues': issues
        }


# ============================================
# USAGE
# ============================================

if __name__ == "__main__":
    # Configure your paths
    PATHS = {
        'cpg': 'data/cpg',      # Adjust to your path
        'input': 'data/input'   # Adjust to your path
    }
    
    checker = CPGPipelineChecker(PATHS)
    
    # Check a specific dataset (e.g., dataset 0)
    results = checker.run_full_check(
        cpg_file='0_cpg.pkl',
        input_file='0_cpg_input.pkl'
    )
    
    # Optional: Check multiple datasets
    print("\n\n" + "="*80)
    print("BATCH CHECK - Multiple Datasets")
    print("="*80)
    
    for i in [0, 1, 2]:  # Check first 3 datasets
        print(f"\n--- Dataset {i} ---")
        try:
            checker.run_full_check(
                cpg_file=f'{i}_cpg.pkl',
                input_file=f'{i}_cpg_input.pkl'
            )
        except FileNotFoundError as e:
            print(f"⚠️ File not found: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")