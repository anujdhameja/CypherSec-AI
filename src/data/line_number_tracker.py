#!/usr/bin/env python3
"""
Line Number Tracker for CPG Processing

This module extracts and tracks line numbers from Joern CPG nodes,
enabling precise mapping of attention weights back to source code lines.

Key Features:
- Extracts line numbers from Joern CPG node properties
- Creates mappings from node_id to line_number
- Handles multiple CPG formats (AST, CFG, PDG)
- Provides line-to-code content mapping
"""

import json
import torch
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data


class LineNumberTracker:
    """
    Extracts and tracks line numbers from Joern CPG nodes.
    Maps AST nodes to their source code line numbers.
    """
    
    @staticmethod
    def extract_line_from_node(node: Dict) -> int:
        """
        Extract line number from Joern CPG node.
        
        Node structure from Joern:
        {
            "id": 123,
            "label": "IDENTIFIER", 
            "properties": {
                "code": "buffer",
                "lineNumber": 4,
                "columnNumber": 10,
                ...
            }
        }
        
        Args:
            node: Joern CPG node dictionary
            
        Returns:
            Line number (int), or -1 if not found
        """
        # Try different possible keys for line number
        if 'properties' in node:
            props = node['properties']
            # Joern uses these keys
            if 'lineNumber' in props:
                return int(props['lineNumber'])
            elif 'LINE_NUMBER' in props:
                return int(props['LINE_NUMBER'])
            elif 'line' in props:
                return int(props['line'])
        
        # Fallback: try direct access
        if 'lineNumber' in node:
            return int(node['lineNumber'])
        if 'line' in node:
            return int(node['line'])
        
        # Unknown line number
        return -1
    
    @staticmethod
    def create_line_mapping(cpg_json: Dict) -> Tuple[Dict[int, int], Dict[int, str]]:
        """
        Create mapping from node_id to line_number and line_content.
        
        Args:
            cpg_json: Parsed JSON from Joern CPG query
            
        Returns:
            node_to_line: Dict[node_id -> line_number]
            line_to_code: Dict[line_number -> source_code]
        """
        node_to_line = {}
        line_to_code = {}
        
        print(f"🔍 Processing CPG JSON for line number extraction...")
        
        # Extract from all graph types (AST, CFG, PDG)
        for graph_type in ['AST', 'CFG', 'PDG', 'nodes']:  # Added 'nodes' for direct format
            if graph_type in cpg_json:
                nodes = cpg_json[graph_type]
                
                # Handle different formats
                if isinstance(nodes, dict) and 'nodes' in nodes:
                    nodes = nodes['nodes']
                elif isinstance(nodes, list):
                    pass  # Already a list of nodes
                else:
                    continue
                
                for node in nodes:
                    node_id = node.get('id')
                    line_num = LineNumberTracker.extract_line_from_node(node)
                    
                    if node_id is not None and line_num >= 0:
                        node_to_line[node_id] = line_num
                        
                        # Store source code if available
                        if 'properties' in node and 'code' in node['properties']:
                            code = node['properties']['code']
                            if line_num not in line_to_code:
                                line_to_code[line_num] = code
                            else:
                                # Combine multiple code fragments on same line
                                existing_code = line_to_code[line_num]
                                if code not in existing_code:
                                    line_to_code[line_num] = f"{existing_code} {code}"
        
        print(f"✅ Extracted {len(node_to_line)} node-to-line mappings")
        print(f"✅ Extracted {len(line_to_code)} line-to-code mappings")
        
        return node_to_line, line_to_code
    
    @staticmethod
    def map_attention_to_lines(attention_weights: torch.Tensor, 
                              line_numbers: torch.Tensor,
                              batch: Optional[torch.Tensor] = None) -> Dict[int, List[float]]:
        """
        Map attention weights back to source code lines.
        
        Args:
            attention_weights: [num_nodes] attention scores
            line_numbers: [num_nodes] line numbers for each node
            batch: Optional batch assignment for multi-graph batches
            
        Returns:
            Dict[line_number -> List[attention_scores]]
        """
        line_attention = {}
        
        # Handle single graph or batch
        if batch is None:
            batch = torch.zeros(len(attention_weights), dtype=torch.long)
        
        # Process each graph in batch
        for graph_idx in range(batch.max().item() + 1):
            graph_mask = (batch == graph_idx)
            graph_attention = attention_weights[graph_mask]
            graph_lines = line_numbers[graph_mask]
            
            # Group attention by line number
            for node_idx, line_num in enumerate(graph_lines):
                line_num = line_num.item()
                if line_num >= 0:  # Valid line number
                    if line_num not in line_attention:
                        line_attention[line_num] = []
                    line_attention[line_num].append(graph_attention[node_idx].item())
        
        return line_attention
    
    @staticmethod
    def aggregate_line_attention(line_attention: Dict[int, List[float]], 
                               method: str = 'mean') -> Dict[int, float]:
        """
        Aggregate multiple attention scores per line.
        
        Args:
            line_attention: Dict[line_number -> List[attention_scores]]
            method: Aggregation method ('mean', 'max', 'sum')
            
        Returns:
            Dict[line_number -> aggregated_attention_score]
        """
        aggregated = {}
        
        for line_num, scores in line_attention.items():
            if method == 'mean':
                aggregated[line_num] = sum(scores) / len(scores)
            elif method == 'max':
                aggregated[line_num] = max(scores)
            elif method == 'sum':
                aggregated[line_num] = sum(scores)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        
        return aggregated


class EnhancedGraphData(Data):
    """
    Enhanced PyTorch Geometric Data object with line number tracking.
    
    This extends the standard PyG Data object to include:
    - Line numbers for each node
    - Source code content
    - Function name and metadata
    - Node ID tracking
    """
    
    def __init__(self, x, edge_index, y, **kwargs):
        super().__init__(x=x, edge_index=edge_index, y=y, **kwargs)
    
    @staticmethod
    def from_cpg_with_lines(cpg_data: Dict, node_embeddings: torch.Tensor,
                           label: int) -> 'EnhancedGraphData':
        """
        Create graph data with line number tracking.
        
        Args:
            cpg_data: Processed CPG dict with node_to_line mapping
            node_embeddings: [num_nodes, embedding_dim] tensor
            label: 0=safe, 1=vulnerable
            
        Returns:
            EnhancedGraphData object with line number tracking
        """
        # Build edge index
        edges = cpg_data.get('edges', [])
        if edges:
            edge_list = [[e['source'], e['target']] for e in edges if 'source' in e and 'target' in e]
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                # Empty graph
                edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create node ID to index mapping
        nodes = cpg_data.get('nodes', [])
        node_ids = [n.get('id', i) for i, n in enumerate(nodes)]
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Map line numbers to node indices
        line_numbers = torch.full((len(node_ids),), -1, dtype=torch.long)
        if 'node_to_line' in cpg_data:
            for node_id, line_num in cpg_data['node_to_line'].items():
                if node_id in id_to_idx:
                    idx = id_to_idx[node_id]
                    line_numbers[idx] = line_num
        
        # Create data object
        data = EnhancedGraphData(
            x=node_embeddings,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            line_numbers=line_numbers,  # CRITICAL: Add line numbers
            source_code=cpg_data.get('source_code', ''),
            function_name=cpg_data.get('function_name', 'unknown'),
            node_ids=torch.tensor(node_ids, dtype=torch.long),
            line_to_code=cpg_data.get('line_to_code', {})
        )
        
        return data
    
    def get_line_attention_mapping(self, attention_weights: torch.Tensor) -> Dict[int, float]:
        """
        Get attention mapping for this graph's lines.
        
        Args:
            attention_weights: [num_nodes] attention scores
            
        Returns:
            Dict[line_number -> attention_score]
        """
        if not hasattr(self, 'line_numbers'):
            return {}
        
        line_attention = LineNumberTracker.map_attention_to_lines(
            attention_weights, self.line_numbers
        )
        
        return LineNumberTracker.aggregate_line_attention(line_attention, method='mean')
    
    def get_source_lines_with_attention(self, attention_weights: torch.Tensor) -> List[Dict]:
        """
        Get source code lines with their attention scores.
        
        Args:
            attention_weights: [num_nodes] attention scores
            
        Returns:
            List of dicts with line info and attention scores
        """
        line_attention = self.get_line_attention_mapping(attention_weights)
        
        # Get source code lines
        if hasattr(self, 'source_code') and self.source_code:
            source_lines = self.source_code.split('\n')
        else:
            source_lines = []
        
        # Combine line content with attention
        result = []
        for line_num, attention_score in sorted(line_attention.items()):
            line_content = ""
            if line_num < len(source_lines):
                line_content = source_lines[line_num]
            elif hasattr(self, 'line_to_code') and line_num in self.line_to_code:
                line_content = self.line_to_code[line_num]
            
            result.append({
                'line_number': line_num,
                'line_content': line_content,
                'attention_score': attention_score,
                'has_source': bool(line_content)
            })
        
        return result


def enhanced_json_process(json_path: str) -> List[Dict]:
    """
    Enhanced CPG JSON processor that preserves line numbers.
    
    This function processes Joern CPG JSON files and extracts line number
    information for precise attention-to-source mapping.
    
    Args:
        json_path: Path to CPG JSON file
        
    Returns:
        List of processed graph dictionaries with line number tracking
    """
    print(f"📄 Processing CPG JSON: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed_graphs = []
    
    # Handle different JSON formats
    if isinstance(data, list):
        functions_data = data
    elif isinstance(data, dict) and 'functions' in data:
        functions_data = data['functions']
    else:
        functions_data = [data]  # Single function
    
    for i, function_data in enumerate(functions_data):
        print(f"  Processing function {i+1}/{len(functions_data)}")
        
        # Extract AST/CFG/PDG
        ast_nodes = function_data.get('AST', {}).get('nodes', [])
        ast_edges = function_data.get('AST', {}).get('edges', [])
        
        # If no AST, try direct nodes/edges
        if not ast_nodes:
            ast_nodes = function_data.get('nodes', [])
            ast_edges = function_data.get('edges', [])
        
        # CRITICAL: Extract line numbers
        tracker = LineNumberTracker()
        node_to_line, line_to_code = tracker.create_line_mapping(function_data)
        
        # Build graph representation
        graph = {
            'nodes': ast_nodes,
            'edges': ast_edges,
            'node_to_line': node_to_line,  # ADD THIS
            'line_to_code': line_to_code,  # ADD THIS
            'function_name': function_data.get('name', f'function_{i}'),
            'source_code': function_data.get('code', function_data.get('source_code', ''))
        }
        
        processed_graphs.append(graph)
    
    print(f"✅ Processed {len(processed_graphs)} functions with line number tracking")
    return processed_graphs


def collate_graphs_with_lines(batch: List[EnhancedGraphData]) -> EnhancedGraphData:
    """
    Custom collate function that preserves line numbers.
    Use this in your DataLoader.
    
    Args:
        batch: List of EnhancedGraphData objects
        
    Returns:
        Batched EnhancedGraphData with preserved line numbers
    """
    from torch_geometric.data import Batch
    
    # Use PyG's Batch but ensure line_numbers are preserved
    batched = Batch.from_data_list(batch)
    
    # Verify line numbers are batched correctly
    assert hasattr(batched, 'line_numbers'), "Line numbers not preserved in batch!"
    
    return batched


def verify_line_tracking(dataset: List[EnhancedGraphData]):
    """
    Verify that line numbers are correctly tracked.
    
    Args:
        dataset: List of EnhancedGraphData objects to verify
    """
    print("=" * 70)
    print("LINE NUMBER TRACKING VERIFICATION")
    print("=" * 70)
    
    issues = []
    
    for i, data in enumerate(dataset):
        # Check if line_numbers attribute exists
        if not hasattr(data, 'line_numbers'):
            issues.append(f"Graph {i}: Missing line_numbers attribute")
            continue
        
        # Check if line numbers are valid
        line_nums = data.line_numbers
        num_valid = (line_nums >= 0).sum().item()
        num_invalid = (line_nums < 0).sum().item()
        
        print(f"\nGraph {i}:")
        print(f"  Total nodes: {len(line_nums)}")
        print(f"  Valid line mappings: {num_valid}")
        print(f"  Invalid line mappings: {num_invalid}")
        
        if num_valid == 0:
            issues.append(f"Graph {i}: No valid line numbers!")
        
        # Check for source code
        if hasattr(data, 'source_code') and data.source_code:
            lines = data.source_code.split('\n')
            max_line = line_nums[line_nums >= 0].max().item() if num_valid > 0 else -1
            if max_line >= len(lines):
                issues.append(f"Graph {i}: Line number {max_line} exceeds source code length {len(lines)}")
        
        # Show sample line mappings
        if num_valid > 0:
            valid_lines = line_nums[line_nums >= 0][:5]  # First 5 valid lines
            print(f"  Sample line numbers: {valid_lines.tolist()}")
    
    print("\n" + "=" * 70)
    if issues:
        print("⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All graphs have valid line number tracking!")
    print("=" * 70)


def test_line_number_tracker():
    """Test the line number tracker with sample data"""
    
    print("🧪 Testing Line Number Tracker")
    print("="*50)
    
    # Sample CPG JSON structure
    sample_cpg = {
        "AST": {
            "nodes": [
                {
                    "id": 1,
                    "label": "METHOD",
                    "properties": {
                        "code": "void test()",
                        "lineNumber": 1
                    }
                },
                {
                    "id": 2,
                    "label": "IDENTIFIER",
                    "properties": {
                        "code": "buffer",
                        "lineNumber": 3
                    }
                },
                {
                    "id": 3,
                    "label": "CALL",
                    "properties": {
                        "code": "strcpy(buffer, input)",
                        "lineNumber": 4
                    }
                }
            ],
            "edges": [
                {"source": 1, "target": 2},
                {"source": 2, "target": 3}
            ]
        },
        "source_code": "void test() {\n  char buffer[64];\n  strcpy(buffer, input);\n}"
    }
    
    # Test line number extraction
    tracker = LineNumberTracker()
    node_to_line, line_to_code = tracker.create_line_mapping(sample_cpg)
    
    print(f"Node to line mapping: {node_to_line}")
    print(f"Line to code mapping: {line_to_code}")
    
    # Test attention mapping
    attention_weights = torch.tensor([0.1, 0.8, 0.9])
    line_numbers = torch.tensor([1, 3, 4])
    
    line_attention = tracker.map_attention_to_lines(attention_weights, line_numbers)
    aggregated = tracker.aggregate_line_attention(line_attention)
    
    print(f"Line attention mapping: {aggregated}")
    
    print("✅ Line number tracker test completed!")


if __name__ == "__main__":
    print("Line Number Tracking Setup for Enhanced Vulnerability Detection")
    print("=" * 70)
    
    # Run test
    test_line_number_tracker()
    
    print("\n📋 Implementation Checklist:\n")
    print("✅ 1. LineNumberTracker - Extract line numbers from Joern CPG")
    print("✅ 2. EnhancedGraphData - PyG Data with line number tracking")
    print("✅ 3. enhanced_json_process - CPG processing with line preservation")
    print("✅ 4. collate_graphs_with_lines - Custom DataLoader collation")
    print("✅ 5. verify_line_tracking - Verification utilities")
    
    print("\n💡 Key Benefits:")
    print("   • Precise mapping of attention weights to source code lines")
    print("   • Support for multiple nodes mapping to same line")
    print("   • Handles various Joern CPG JSON formats")
    print("   • Preserves line information through PyG batching")
    
    print("\n🎯 Result:")
    print("   Attention weights can now be precisely mapped to source lines!")
    print("   Perfect integration with our pattern-enhanced detection system!")