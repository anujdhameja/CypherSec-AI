import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

class CPGStructureValidator:
    def __init__(self, cpg_dir: str):
        """Initialize the validator with the directory containing CPG files.
        
        Args:
            cpg_dir: Path to the directory containing .pkl files
        """
        self.cpg_dir = Path(cpg_dir)
        self.results = []
        
    def validate_single_file(self, file_path: Path) -> Dict:
        """Validate the structure of a single .pkl file.
        
        Args:
            file_path: Path to the .pkl file to validate
            
        Returns:
            Dictionary containing validation results
        """
        result = {
            'file': str(file_path),
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {
                'total_rows': 0,
                'valid_rows': 0,
                'nodes_count': 0,
                'edges_count': 0,
                'invalid_rows': 0
            }
        }
        
        try:
            # Load the pickle file
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                
            # Check if data is a pandas DataFrame
            if not isinstance(df, pd.DataFrame):
                result['is_valid'] = False
                result['errors'].append(f"Expected pandas DataFrame, got {type(df).__name__}")
                return result
                
            # Check required columns
            required_columns = {'target', 'func', 'Index', 'cpg'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                result['is_valid'] = False
                result['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
                return result
                
            result['stats']['total_rows'] = len(df)
            
            # Check each row in the DataFrame
            for idx, row in df.iterrows():
                row_valid = True
                
                # Check if 'cpg' is a dictionary with required keys
                if not isinstance(row['cpg'], dict):
                    result['warnings'].append(f"Row {idx}: 'cpg' is not a dictionary")
                    result['stats']['invalid_rows'] += 1
                    continue
                    
                # Check for required keys in cpg dict
                if 'nodes' not in row['cpg'] or 'edges' not in row['cpg']:
                    result['warnings'].append(f"Row {idx}: 'cpg' is missing 'nodes' or 'edges' key")
                    result['stats']['invalid_rows'] += 1
                    continue
                    
                # Validate nodes
                nodes = row['cpg'].get('nodes', [])
                if not isinstance(nodes, list):
                    result['warnings'].append(f"Row {idx}: 'nodes' is not a list")
                    row_valid = False
                else:
                    result['stats']['nodes_count'] += len(nodes)
                    # Check each node
                    for node_idx, node in enumerate(nodes):
                        if not isinstance(node, dict):
                            result['warnings'].append(
                                f"Row {idx}, node {node_idx}: Node is not a dictionary"
                            )
                            row_valid = False
                            continue
                            
                        # Check required node fields
                        for field in ['id', 'label', 'code']:
                            if field not in node:
                                result['warnings'].append(
                                    f"Row {idx}, node {node_idx}: Missing required field '{field}'"
                                )
                                row_valid = False
                
                # Validate edges
                edges = row['cpg'].get('edges', [])
                if not isinstance(edges, list):
                    result['warnings'].append(f"Row {idx}: 'edges' is not a list")
                    row_valid = False
                else:
                    result['stats']['edges_count'] += len(edges)
                    # Check each edge
                    for edge_idx, edge in enumerate(edges):
                        if not isinstance(edge, dict):
                            result['warnings'].append(
                                f"Row {idx}, edge {edge_idx}: Edge is not a dictionary"
                            )
                            row_valid = False
                            continue
                            
                        # Check required edge fields
                        for field in ['source', 'target', 'label']:
                            if field not in edge:
                                result['warnings'].append(
                                    f"Row {idx}, edge {edge_idx}: Missing required field '{field}'"
                                )
                                row_valid = False
                        
                        # Check if source and target nodes exist
                        if 'nodes' in locals() and isinstance(nodes, list):
                            node_ids = {node['id'] for node in nodes if isinstance(node, dict) and 'id' in node}
                            if 'source' in edge and edge['source'] not in node_ids:
                                result['warnings'].append(
                                    f"Row {idx}, edge {edge_idx}: Source node {edge['source']} not found in nodes"
                                )
                                row_valid = False
                            if 'target' in edge and edge['target'] not in node_ids:
                                result['warnings'].append(
                                    f"Row {idx}, edge {edge_idx}: Target node {edge['target']} not found in nodes"
                                )
                                row_valid = False
                
                if row_valid:
                    result['stats']['valid_rows'] += 1
                else:
                    result['stats']['invalid_rows'] += 1
            
            # Calculate validity based on number of valid rows
            if result['stats']['total_rows'] > 0:
                validity_ratio = result['stats']['valid_rows'] / result['stats']['total_rows']
                result['is_valid'] = validity_ratio >= 0.5  # Consider file valid if at least 50% of rows are valid
            
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Error processing file: {str(e)}")
            
        return result
    
    def validate_all_files(self) -> List[Dict]:
        """Validate all .pkl files in the CPG directory.
        
        Returns:
            List of validation results for each file
        """
        if not self.cpg_dir.exists() or not self.cpg_dir.is_dir():
            raise ValueError(f"Directory not found: {self.cpg_dir}")
            
        pkl_files = list(self.cpg_dir.glob('*.pkl'))
        if not pkl_files:
            print(f"No .pkl files found in {self.cpg_dir}")
            return []
            
        print(f"Found {len(pkl_files)} .pkl files to validate...")
        
        results = []
        for pkl_file in tqdm(pkl_files, desc="Validating files"):
            result = self.validate_single_file(pkl_file)
            results.append(result)
            
            # Print summary for this file
            status = "PASSED" if result['is_valid'] else "FAILED"
            stats = result['stats']
            print(f"\n{pkl_file.name}: {status}")
            print(f"  Rows: {stats['total_rows']} total, {stats['valid_rows']} valid, {stats['invalid_rows']} invalid")
            print(f"  Nodes: {stats['nodes_count']}, Edges: {stats['edges_count']}")
            
            if result['errors']:
                print(f"  Errors: {len(result['errors'])}")
            if result['warnings']:
                print(f"  Warnings: {len(result['warnings'])}")
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str = None) -> str:
        """Generate a summary report of the validation results.
        
        Args:
            results: List of validation results
            output_file: Optional path to save the report
            
        Returns:
            The report as a string
        """
        if not results:
            return "No validation results to report."
            
        total_files = len(results)
        valid_files = sum(1 for r in results if r['is_valid'])
        total_errors = sum(len(r.get('errors', [])) for r in results)
        total_warnings = sum(len(r.get('warnings', [])) for r in results)
        total_rows = sum(r['stats'].get('total_rows', 0) for r in results)
        valid_rows = sum(r['stats'].get('valid_rows', 0) for r in results)
        total_nodes = sum(r['stats'].get('nodes_count', 0) for r in results)
        total_edges = sum(r['stats'].get('edges_count', 0) for r in results)
        
        report = [
            "=" * 80,
            "CPG Structure Validation Report",
            "=" * 80,
            f"Total files checked: {total_files}",
            f"Valid files: {valid_files} ({(valid_files/total_files*100):.1f}%)",
            f"Total rows: {total_rows}",
            f"Valid rows: {valid_rows} ({(valid_rows/max(1, total_rows)*100):.1f}%)",
            f"Total nodes: {total_nodes}",
            f"Total edges: {total_edges}",
            f"Total errors: {total_errors}",
            f"Total warnings: {total_warnings}",
            "=" * 80,
            ""
        ]
        
        # Add detailed information for files with issues
        for result in results:
            if not result['is_valid'] or result.get('warnings') or result.get('errors'):
                report.append(f"\nFile: {result['file']}")
                report.append("-" * 80)
                
                stats = result['stats']
                report.append(f"Status: {'VALID' if result['is_valid'] else 'INVALID'}")
                report.append(f"Rows: {stats['total_rows']} total, {stats['valid_rows']} valid, {stats['invalid_rows']} invalid")
                report.append(f"Nodes: {stats['nodes_count']}, Edges: {stats['edges_count']}")
                
                if result.get('errors'):
                    report.append("\nErrors:")
                    for error in result['errors'][:5]:  # Show first 5 errors
                        report.append(f"  - {error}")
                    if len(result['errors']) > 5:
                        report.append(f"  ... and {len(result['errors']) - 5} more errors")
                        
                if result.get('warnings'):
                    report.append("\nWarnings:")
                    for warning in result['warnings'][:5]:  # Show first 5 warnings
                        report.append(f"  - {warning}")
                    if len(result['warnings']) > 5:
                        report.append(f"  ... and {len(result['warnings']) - 5} more warnings")
                
                report.append("")
        
        report = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nReport saved to {output_file}")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CPG structure in .pkl files')
    parser.add_argument('cpg_dir', type=str, help='Directory containing .pkl files')
    parser.add_argument('--output', '-o', type=str, help='Output file for the report')
    
    args = parser.parse_args()
    
    try:
        validator = CPGStructureValidator(args.cpg_dir)
        results = validator.validate_all_files()
        
        if results:
            report = validator.generate_report(results, args.output)
            print("\n" + "=" * 80)
            print("Validation complete!")
            print("=" * 80)
            
            # Print a quick summary
            valid_count = sum(1 for r in results if r['is_valid'])
            print(f"\nSummary: {valid_count}/{len(results)} files passed validation")
            
            if valid_count < len(results):
                print("\nFor detailed error information, check the report above")
                if args.output:
                    print(f"Full report saved to: {args.output}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
