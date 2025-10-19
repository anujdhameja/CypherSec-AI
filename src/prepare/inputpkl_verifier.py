"""
Input PKL Verifier - Displays and analyzes input pickle files in a readable format
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from tabulate import tabulate
from pathlib import Path
from collections import Counter
import torch
from torch_geometric.data import Data

def print_section(title, width=80):
    """Print a section header"""
    print("\n" + "=" * width)
    print(f"{title.upper()}".center(width))
    print("=" * width)

def display_dataframe_info(df):
    """Display detailed information about the DataFrame"""
    # Basic info
    print_section("dataframe overview")
    print(f"Shape: {df.shape} (rows, columns)")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Memory usage
    print("\nMemory Usage:")
    print(f"Total: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    
    # Data types
    print("\nData Types:")
    print(df.dtypes)
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing Values:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found.")

def display_graph_info(graph, graph_idx=0):
    """Display information about a PyG Data graph"""
    print_section(f"graph {graph_idx} details")
    
    if not isinstance(graph, Data):
        print(f"⚠️ Not a PyG Data object: {type(graph)}")
        return
    
    # Basic info
    print(f"Graph Type: {type(graph).__name__}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Number of node features: {graph.num_node_features}")
    
    # Check for common attributes
    common_attrs = ['y', 'edge_attr', 'batch', 'pos']
    for attr in common_attrs:
        if hasattr(graph, attr):
            val = getattr(graph, attr)
            if val is not None:
                print(f"{attr}: {type(val).__name__} {tuple(val.shape) if hasattr(val, 'shape') else ''}")
    
    # Edge index info
    if hasattr(graph, 'edge_index') and graph.edge_index is not None:
        print("\nEdge Index:")
        print(f"  Shape: {tuple(graph.edge_index.shape)}")
        print(f"  Min index: {graph.edge_index.min().item()}")
        print(f"  Max index: {graph.edge_index.max().item()}")
        
        # Check for self-loops
        if graph.edge_index.shape[1] > 0:
            self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum().item()
            print(f"  Self-loops: {self_loops} ({self_loops/graph.edge_index.shape[1]:.1%})")
    
    # Node features info
    if hasattr(graph, 'x') and graph.x is not None:
        print("\nNode Features:")
        print(f"  Shape: {tuple(graph.x.shape)}")
        print(f"  Type: {graph.x.dtype}")
        
        # Basic statistics if features are numeric
        if graph.x.is_floating_point():
            x_np = graph.x.numpy()
            print(f"  Min: {x_np.min():.4f}")
            print(f"  Max: {x_np.max():.4f}")
            print(f"  Mean: {x_np.mean():.4f} ± {x_np.std():.4f}")
            
            # Check for NaNs/Infs
            nan_count = np.isnan(x_np).sum()
            inf_count = np.isinf(x_np).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  ⚠️ Contains {nan_count} NaNs and {inf_count} Infs")
    
    # Check for graph-level attributes
    print("\nGraph Attributes:")
    if hasattr(graph, 'keys'):
        for key in graph.keys():
            if key not in ['x', 'edge_index', 'y', 'edge_attr', 'batch', 'pos']:
                val = getattr(graph, key, None)
                if val is not None:
                    if isinstance(val, (int, float, str, bool)) or \
                       (hasattr(val, 'shape') and len(val.shape) == 0):
                        print(f"  {key}: {val}")
                    else:
                        print(f"  {key}: {type(val).__name__} {tuple(val.shape) if hasattr(val, 'shape') else ''}")

def display_sample_data(df, sample_size=3):
    """Display sample rows from the DataFrame"""
    print_section(f"sample data (first {min(sample_size, len(df))} rows)")
    
    # Create a copy to avoid modifying the original
    display_df = df.head(sample_size).copy()
    
    # Replace graph objects with summary strings
    for col in display_df.columns:
        if isinstance(display_df[col].iloc[0], (Data, torch.Tensor, np.ndarray)):
            display_df[col] = display_df[col].apply(
                lambda x: f"{type(x).__name__} {tuple(x.shape) if hasattr(x, 'shape') else ''}"
            )
        elif isinstance(display_df[col].iloc[0], (list, dict)):
            display_df[col] = display_df[col].apply(
                lambda x: f"{type(x).__name__} (len={len(x)})"
            )
    
    # Display the table
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=True))

def analyze_target_distribution(df, target_col='target'):
    """Analyze the distribution of target values"""
    if target_col not in df.columns:
        return
        
    print_section("target distribution")
    target_counts = df[target_col].value_counts().sort_index()
    target_percent = df[target_col].value_counts(normalize=True).sort_index() * 100
    
    dist_df = pd.DataFrame({
        'Count': target_counts,
        'Percentage': [f"{p:.1f}%" for p in target_percent]
    })
    
    print(tabulate(dist_df, headers='keys', tablefmt='grid', showindex=True))
    
    # Check for class imbalance
    if len(target_counts) > 1:
        imbalance_ratio = target_counts.max() / target_counts.min()
        if imbalance_ratio > 2:
            print(f"\n⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")

def verify_input_pkl(file_path, max_graphs_to_analyze=3):
    """Main function to verify an input pickle file"""
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return
    
    print_section(f"verifying: {file_path}")
    
    try:
        # Load the data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            print("✓ Loaded DataFrame successfully")
            df = data
            
            # Display DataFrame info and samples
            display_dataframe_info(df)
            display_sample_data(df)
            analyze_target_distribution(df)
            
            # Check for graph data in columns
            graph_cols = [col for col in df.columns if isinstance(df[col].iloc[0], Data)]
            
            if graph_cols:
                print_section("graph data analysis")
                print(f"Found graph data in columns: {', '.join(graph_cols)}")
                
                # Analyze first few graphs in detail
                for col in graph_cols[:2]:  # Limit to first 2 graph columns
                    print(f"\nAnalyzing graphs in column: {col}")
                    for i in range(min(max_graphs_to_analyze, len(df))):
                        display_graph_info(df[col].iloc[i], i)
            
            # Additional analysis for other columns
            print_section("column analysis")
            for col in df.columns:
                if col not in graph_cols and col != 'target':
                    print(f"\nColumn: {col}")
                    print(f"Type: {df[col].dtype}")
                    print(f"Unique values: {df[col].nunique()}")
                    if df[col].dtype in ['object', 'category']:
                        print("Most common values:")
                        print(df[col].value_counts().head(5))
                    elif np.issubdtype(df[col].dtype, np.number):
                        print(f"Min: {df[col].min()}")
                        print(f"Max: {df[col].max()}")
                        print(f"Mean: {df[col].mean():.2f}")
                        print(f"Std: {df[col].std():.2f}")
        
        elif isinstance(data, (Data, dict, list)):
            print("✓ Loaded data successfully")
            if isinstance(data, Data):
                display_graph_info(data, 0)
            elif isinstance(data, dict):
                print("\nDictionary contents:")
                for key, value in data.items():
                    if isinstance(value, (Data, torch.Tensor, np.ndarray)):
                        print(f"{key}: {type(value).__name__} {tuple(value.shape) if hasattr(value, 'shape') else ''}")
                    else:
                        print(f"{key}: {type(value).__name__} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
            elif isinstance(data, list):
                print(f"\nList length: {len(data)}")
                if len(data) > 0:
                    print("\nFirst item type:", type(data[0]))
                    if isinstance(data[0], Data):
                        display_graph_info(data[0], 0)
        else:
            print(f"Unsupported data type: {type(data)}")
    
    except Exception as e:
        print(f"❌ Error loading/processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to handle command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify and analyze input pickle files')
    parser.add_argument('file_path', type=str, help='Path to the input pickle file')
    parser.add_argument('--max-graphs', type=int, default=3, 
                        help='Maximum number of graphs to analyze in detail (default: 3)')
    
    args = parser.parse_args()
    
    # Verify the input file
    verify_input_pkl(args.file_path, args.max_graphs)

if __name__ == "__main__":
    main()
