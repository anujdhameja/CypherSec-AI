import os
import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

def check_edge_structure(cpg_dir, sample_size=5):
    """Check the structure of edges across multiple CPG files.
    
    Args:
        cpg_dir: Directory containing CPG .pkl files
        sample_size: Number of files to sample for checking
    """
    cpg_dir = Path(cpg_dir)
    pkl_files = list(cpg_dir.glob('*.pkl'))
    
    if not pkl_files:
        print(f"No .pkl files found in {cpg_dir}")
        return
    
    print(f"Found {len(pkl_files)} .pkl files. Checking edge structure in {min(sample_size, len(pkl_files))} files...")
    print("-" * 80)
    
    # Track all unique edge structures
    edge_structures = defaultdict(int)
    edge_labels = set()
    
    for i, pkl_file in enumerate(pkl_files[:sample_size]):
        try:
            with open(pkl_file, 'rb') as f:
                df = pickle.load(f)
                
            print(f"\nFile: {pkl_file.name}")
            print("-" * 40)
            
            # Check first few rows
            for _, row in df.head(2).iterrows():
                if 'cpg' in row and isinstance(row['cpg'], dict) and 'edges' in row['cpg']:
                    edges = row['cpg']['edges']
                    if not edges:
                        print("  No edges found in this row")
                        continue
                        
                    # Check structure of first few edges
                    print(f"  Found {len(edges)} edges")
                    print("  Sample edge structure:")
                    for j, edge in enumerate(edges[:2]):  # Show first 2 edges
                        print(f"  Edge {j+1}:")
                        print(f"    Type: {type(edge).__name__}")
                        if isinstance(edge, dict):
                            print(f"    Keys: {list(edge.keys())}")
                            edge_labels.add(edge.get('label', 'N/A'))
                            
                            # Track this edge's structure
                            structure = tuple(sorted(edge.keys()))
                            edge_structures[structure] += 1
                            
                            # Show values for the first edge
                            if j == 0:
                                print("    Values:")
                                for k, v in edge.items():
                                    print(f"      {k}: {type(v).__name__} = {v}")
                        else:
                            print(f"    Value: {edge}")
                    
                    # Check consistency of edge structures in this row
                    if edges:
                        first_structure = tuple(sorted(edges[0].keys()))
                        consistent = all(tuple(sorted(e.keys())) == first_structure for e in edges)
                        print(f"  All edges have same structure: {consistent}")
                        if not consistent:
                            print("  Warning: Inconsistent edge structures found in this row!")
                
        except Exception as e:
            print(f"  Error processing {pkl_file.name}: {str(e)}")
    
    # Print summary of edge structures found
    print("\n" + "=" * 80)
    print("Edge Structure Summary:")
    print("-" * 80)
    print(f"Total unique edge structures: {len(edge_structures)}")
    
    if edge_structures:
        print("\nEdge structures found (format: (field1, field2, ...): count):")
        for structure, count in edge_structures.items():
            print(f"  {structure}: {count} edges")
    
    if edge_labels:
        print(f"\nEdge labels found: {sorted(edge_labels)}")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_edge_structure.py <path_to_cpg_dir> [sample_size]")
        sys.exit(1)
    
    cpg_dir = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    check_edge_structure(cpg_dir, sample_size)
