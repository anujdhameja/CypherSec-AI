import os
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class EdgeStructureChecker:
    def __init__(self, cpg_dir: str):
        """Initialize the EdgeStructureChecker with the directory containing CPG files.
        
        Args:
            cpg_dir: Path to the directory containing .pkl files
        """
        self.cpg_dir = Path(cpg_dir)
        self.results = []
        
    def check_single_file(self, file_path: Path) -> Tuple[bool, Dict]:
        """Check the structure of a single .pkl file.
        
        Args:
            file_path: Path to the .pkl file to check
            
        Returns:
            Tuple of (is_valid, result_dict) where is_valid is a boolean indicating
            if the structure is valid, and result_dict contains detailed information
            about the validation.
        """
        result = {
            'file': str(file_path),
            'is_valid': True,
            'errors': [],
            'function_count': 0,
            'node_count': 0,
            'edge_count': 0,
            'missing_fields': [],
            'type_errors': []
        }
        
        try:
            # Load the pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # Check if data is a pandas DataFrame
            if not isinstance(data, pd.DataFrame):
                result['is_valid'] = False
                result['errors'].append(f"Expected pandas DataFrame, got {type(data).__name__}")
                return False, result
                
            # Check if 'cpg' column exists
            if 'cpg' not in data.columns:
                result['is_valid'] = False
                result['errors'].append("Missing 'cpg' column in DataFrame")
                return False, result
                
            # Check each CPG in the DataFrame
            for idx, cpg_data in enumerate(data['cpg']):
                if not isinstance(cpg_data, dict):
                    result['errors'].append(f"Row {idx}: Expected dict in 'cpg' column, got {type(cpg_data).__name__}")
                    result['is_valid'] = False
                    continue
                    
                # Check if 'functions' key exists
                if 'functions' not in cpg_data:
                    result['errors'].append(f"Row {idx}: Missing 'functions' key in CPG data")
                    result['is_valid'] = False
                    continue
                    
                # Check each function in the functions list
                for func_idx, func in enumerate(cpg_data['functions']):
                    result['function_count'] += 1
                    
                    # Check required top-level fields
                    for field in ['function', 'file', 'nodes', 'edges']:
                        if field not in func:
                            result['missing_fields'].append(f"Row {idx}, function {func_idx}: Missing required field '{field}'")
                            result['is_valid'] = False
                    
                    # Check nodes
                    if 'nodes' in func:
                        result['node_count'] += len(func['nodes'])
                        for node_idx, node in enumerate(func['nodes']):
                            for field in ['id', 'label', 'code']:
                                if field not in node:
                                    result['missing_fields'].append(
                                        f"Row {idx}, function {func_idx}, node {node_idx}: Missing node field '{field}'"
                                    )
                                    result['is_valid'] = False
                    
                    # Check edges
                    if 'edges' in func:
                        result['edge_count'] += len(func['edges'])
                        for edge_idx, edge in enumerate(func['edges']):
                            for field in ['source', 'target', 'label']:
                                if field not in edge:
                                    result['missing_fields'].append(
                                        f"Row {idx}, function {func_idx}, edge {edge_idx}: Missing edge field '{field}'"
                                    )
                                    result['is_valid'] = False
                            
                            # Check if source and target exist in nodes
                            if 'nodes' in func and 'source' in edge and 'target' in edge:
                                node_ids = {node['id'] for node in func['nodes']}
                                if edge['source'] not in node_ids:
                                    result['errors'].append(
                                        f"Row {idx}, function {func_idx}, edge {edge_idx}: "
                                        f"Source node {edge['source']} not found in nodes"
                                    )
                                    result['is_valid'] = False
                                if edge['target'] not in node_ids:
                                    result['errors'].append(
                                        f"Row {idx}, function {func_idx}, edge {edge_idx}: "
                                        f"Target node {edge['target']} not found in nodes"
                                    )
                                    result['is_valid'] = False
                                    
        except Exception as e:
            result['is_valid'] = False
            result['errors'].append(f"Error processing file: {str(e)}")
            
        return result['is_valid'], result
    
    def check_all_files(self) -> List[Dict]:
        """Check all .pkl files in the CPG directory.
        
        Returns:
            List of result dictionaries for each file checked.
        """
        if not self.cpg_dir.exists() or not self.cpg_dir.is_dir():
            raise ValueError(f"Directory not found: {self.cpg_dir}")
            
        pkl_files = list(self.cpg_dir.glob('*.pkl'))
        if not pkl_files:
            print(f"No .pkl files found in {self.cpg_dir}")
            return []
            
        print(f"Found {len(pkl_files)} .pkl files to check...")
        
        results = []
        for pkl_file in pkl_files:
            print(f"Checking {pkl_file.name}...")
            is_valid, result = self.check_single_file(pkl_file)
            results.append(result)
            
            # Print summary for this file
            status = "PASSED" if is_valid else "FAILED"
            print(f"  - {status}: {result['function_count']} functions, "
                  f"{result['node_count']} nodes, {result['edge_count']} edges")
            
            if not is_valid:
                print(f"  - Errors: {len(result.get('errors', []))}")
                print(f"  - Missing fields: {len(result.get('missing_fields', []))}")
        
        return results
    
    def generate_report(self, results: List[Dict], output_file: str = None) -> str:
        """Generate a summary report of the validation results.
        
        Args:
            results: List of result dictionaries from check_all_files()
            output_file: Optional path to save the report
            
        Returns:
            The report as a string
        """
        total_files = len(results)
        valid_files = sum(1 for r in results if r['is_valid'])
        total_errors = sum(len(r.get('errors', [])) for r in results)
        total_missing = sum(len(r.get('missing_fields', [])) for r in results)
        total_functions = sum(r.get('function_count', 0) for r in results)
        total_nodes = sum(r.get('node_count', 0) for r in results)
        total_edges = sum(r.get('edge_count', 0) for r in results)
        
        report = [
            "=" * 80,
            "CPG Edge Structure Validation Report",
            "=" * 80,
            f"Total files checked: {total_files}",
            f"Valid files: {valid_files} ({(valid_files/total_files*100):.1f}%)",
            f"Total functions: {total_functions}",
            f"Total nodes: {total_nodes}",
            f"Total edges: {total_edges}",
            f"Total errors: {total_errors}",
            f"Total missing fields: {total_missing}",
            "=" * 80,
            ""
        ]
        
        # Add detailed error information for failed files
        for result in results:
            if not result['is_valid']:
                report.append(f"\nFile: {result['file']}")
                report.append("-" * 40)
                
                if result.get('errors'):
                    report.append("\nErrors:")
                    for error in result['errors']:
                        report.append(f"  - {error}")
                        
                if result.get('missing_fields'):
                    report.append("\nMissing Fields:")
                    for i, field in enumerate(result['missing_fields'][:5], 1):
                        report.append(f"  {i}. {field}")
                    if len(result['missing_fields']) > 5:
                        report.append(f"  ... and {len(result['missing_fields']) - 5} more")
                
                report.append("")
        
        report = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"\nReport saved to {output_file}")
        
        return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CPG edge structure in .pkl files')
    parser.add_argument('cpg_dir', type=str, help='Directory containing .pkl files')
    parser.add_argument('--output', '-o', type=str, help='Output file for the report')
    
    args = parser.parse_args()
    
    try:
        checker = EdgeStructureChecker(args.cpg_dir)
        results = checker.check_all_files()
        report = checker.generate_report(results, args.output)
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
