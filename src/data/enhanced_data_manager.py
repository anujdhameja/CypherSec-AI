#!/usr/bin/env python3
"""
Enhanced Data Manager with Line Number Tracking

This module provides enhanced data management capabilities that integrate
line number tracking with our existing vulnerability detection pipeline.

Key Features:
- Creates EnhancedGraphData objects with line number tracking
- Integrates with existing Word2Vec embeddings
- Supports both synthetic and real CPG data
- Maintains compatibility with existing training pipeline
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec

from src.data.line_number_tracker import EnhancedGraphData, LineNumberTracker, collate_graphs_with_lines


class EnhancedVulnerabilityDataset(Dataset):
    """
    Enhanced dataset that creates graphs with line number tracking.
    
    This dataset can work with:
    1. Real CPG data from Joern (with actual line numbers)
    2. Synthetic data (with simulated line numbers)
    3. Existing pickle files (with enhanced processing)
    """
    
    def __init__(self, data_source, w2v_model: Word2Vec, embedding_dim: int = 100):
        """
        Initialize enhanced dataset.
        
        Args:
            data_source: Can be:
                - List of CPG dictionaries (from enhanced_json_process)
                - Path to pickle file
                - List of (graph_data, labels, node_labels, source_code) tuples
            w2v_model: Word2Vec model for embeddings
            embedding_dim: Embedding dimension
        """
        self.w2v_model = w2v_model
        self.embedding_dim = embedding_dim
        self.data = []
        
        print(f"🔧 Initializing Enhanced Vulnerability Dataset...")
        
        if isinstance(data_source, str):
            # Load from pickle file
            self._load_from_pickle(data_source)
        elif isinstance(data_source, list):
            if len(data_source) > 0 and isinstance(data_source[0], dict):
                # CPG data format
                self._load_from_cpg_data(data_source)
            else:
                # Tuple format (graph_data, labels, node_labels, source_code)
                self._load_from_tuples(data_source)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")
        
        print(f"✅ Dataset initialized with {len(self.data)} samples")
    
    def _load_from_pickle(self, pickle_path: str):
        """Load data from existing pickle file and enhance with line tracking"""
        import pandas as pd
        
        print(f"📄 Loading data from pickle: {pickle_path}")
        df = pd.read_pickle(pickle_path)
        
        for idx, row in df.iterrows():
            graph_data = row['input']
            label = int(row['target'])
            
            # Create enhanced graph data with simulated line numbers
            enhanced_data = self._create_enhanced_from_existing(graph_data, label, idx)
            self.data.append(enhanced_data)
    
    def _load_from_cpg_data(self, cpg_data: List[Dict]):
        """Load data from CPG dictionaries with real line numbers"""
        
        print(f"📊 Processing {len(cpg_data)} CPG functions...")
        
        for cpg in cpg_data:
            # Create node embeddings
            embeddings = self._create_node_embeddings(cpg['nodes'])
            
            # Get label (assume safe=0, vulnerable=1 for now)
            label = cpg.get('target', cpg.get('label', 0))
            
            # Create enhanced graph data
            enhanced_data = EnhancedGraphData.from_cpg_with_lines(cpg, embeddings, label)
            self.data.append(enhanced_data)
    
    def _load_from_tuples(self, tuple_data: List[Tuple]):
        """Load data from tuple format"""
        
        print(f"📋 Processing {len(tuple_data)} tuple samples...")
        
        for i, (graph_data, label, node_labels, source_code) in enumerate(tuple_data):
            # Create enhanced version
            enhanced_data = self._create_enhanced_from_components(
                graph_data, label, node_labels, source_code, i
            )
            self.data.append(enhanced_data)
    
    def _create_node_embeddings(self, nodes: List[Dict]) -> torch.Tensor:
        """Create node embeddings from CPG nodes"""
        
        embeddings = []
        
        for node in nodes:
            # Get code token for embedding
            code = node.get('properties', {}).get('code', '')
            
            # Create embedding
            if code and code in self.w2v_model.wv:
                emb = self.w2v_model.wv[code]
            else:
                emb = np.zeros(self.embedding_dim)
            
            embeddings.append(emb)
        
        return torch.tensor(embeddings, dtype=torch.float32)
    
    def _create_enhanced_from_existing(self, graph_data, label: int, sample_idx: int) -> EnhancedGraphData:
        """Create enhanced graph data from existing PyG Data object"""
        
        # Extract basic info
        x = graph_data.x
        edge_index = graph_data.edge_index
        num_nodes = x.size(0)
        
        # Create simulated line numbers (for existing data without real line info)
        line_numbers = torch.arange(num_nodes, dtype=torch.long)
        
        # Create simulated source code
        source_code = f"// Function {sample_idx}\n"
        for i in range(num_nodes):
            source_code += f"line_{i}(); // Node {i}\n"
        
        # Create enhanced data
        enhanced_data = EnhancedGraphData(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            line_numbers=line_numbers,
            source_code=source_code,
            function_name=f"function_{sample_idx}",
            node_ids=torch.arange(num_nodes, dtype=torch.long),
            line_to_code={i: f"line_{i}();" for i in range(num_nodes)}
        )
        
        return enhanced_data
    
    def _create_enhanced_from_components(self, graph_data, label: int, 
                                       node_labels: List[str], source_code: str, 
                                       sample_idx: int) -> EnhancedGraphData:
        """Create enhanced graph data from individual components"""
        
        # Extract basic info
        x = graph_data.x
        edge_index = graph_data.edge_index
        num_nodes = x.size(0)
        
        # Create line numbers from node labels
        line_numbers = torch.arange(min(num_nodes, len(node_labels)), dtype=torch.long)
        if num_nodes > len(node_labels):
            # Pad with -1 for nodes without labels
            padding = torch.full((num_nodes - len(node_labels),), -1, dtype=torch.long)
            line_numbers = torch.cat([line_numbers, padding])
        
        # Create line to code mapping
        line_to_code = {}
        for i, label_text in enumerate(node_labels):
            if i < num_nodes:
                line_to_code[i] = label_text
        
        # Create enhanced data
        enhanced_data = EnhancedGraphData(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long),
            line_numbers=line_numbers,
            source_code=source_code,
            function_name=f"function_{sample_idx}",
            node_ids=torch.arange(num_nodes, dtype=torch.long),
            line_to_code=line_to_code
        )
        
        return enhanced_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_sample_with_lines(self, idx: int) -> Dict:
        """Get a sample with detailed line information"""
        
        data = self.data[idx]
        
        # Create dummy attention for demonstration
        dummy_attention = torch.rand(data.x.size(0))
        
        # Get line mapping
        line_info = data.get_source_lines_with_attention(dummy_attention)
        
        return {
            'data': data,
            'line_info': line_info,
            'function_name': getattr(data, 'function_name', 'unknown'),
            'source_code': getattr(data, 'source_code', ''),
            'label': data.y.item()
        }


def create_enhanced_data_loader(dataset: EnhancedVulnerabilityDataset, 
                              batch_size: int = 8, 
                              shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader with enhanced graph data and line number tracking.
    
    Args:
        dataset: EnhancedVulnerabilityDataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader with custom collation for line number preservation
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graphs_with_lines  # IMPORTANT: Preserves line numbers
    )


def create_enhanced_dataset_from_existing_data(existing_data_path: str, 
                                             w2v_model: Word2Vec) -> EnhancedVulnerabilityDataset:
    """
    Create enhanced dataset from existing pickle data.
    
    Args:
        existing_data_path: Path to existing pickle file
        w2v_model: Word2Vec model
        
    Returns:
        EnhancedVulnerabilityDataset with line number tracking
    """
    return EnhancedVulnerabilityDataset(existing_data_path, w2v_model)


def create_enhanced_dataset_from_synthetic_data(synthetic_samples: List[Tuple], 
                                              w2v_model: Word2Vec) -> EnhancedVulnerabilityDataset:
    """
    Create enhanced dataset from synthetic data samples.
    
    Args:
        synthetic_samples: List of (graph_data, label, node_labels, source_code) tuples
        w2v_model: Word2Vec model
        
    Returns:
        EnhancedVulnerabilityDataset with line number tracking
    """
    return EnhancedVulnerabilityDataset(synthetic_samples, w2v_model)


def test_enhanced_data_manager():
    """Test the enhanced data manager with sample data"""
    
    print("🧪 Testing Enhanced Data Manager")
    print("="*60)
    
    # Create dummy Word2Vec model
    from gensim.models import Word2Vec
    sentences = [['test', 'code', 'buffer', 'strcpy']]
    w2v_model = Word2Vec(sentences, vector_size=100, min_count=1)
    
    # Create sample synthetic data
    from torch_geometric.data import Data
    
    sample_data = []
    for i in range(3):
        # Create sample graph
        x = torch.randn(5, 100)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        graph_data = Data(x=x, edge_index=edge_index)
        
        # Create sample labels and source
        node_labels = [f"line_{j}" for j in range(5)]
        source_code = f"void function_{i}() {{\n" + "\n".join([f"  {label};" for label in node_labels]) + "\n}"
        label = i % 2  # Alternate between safe and vulnerable
        
        sample_data.append((graph_data, label, node_labels, source_code))
    
    # Create enhanced dataset
    dataset = create_enhanced_dataset_from_synthetic_data(sample_data, w2v_model)
    
    print(f"✅ Created dataset with {len(dataset)} samples")
    
    # Test data loader
    data_loader = create_enhanced_data_loader(dataset, batch_size=2)
    
    print(f"✅ Created data loader with batch_size=2")
    
    # Test batch processing
    for batch_idx, batch in enumerate(data_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch size: {batch.num_graphs}")
        print(f"  Total nodes: {batch.num_nodes}")
        print(f"  Line numbers shape: {batch.line_numbers.shape}")
        print(f"  Has source code: {hasattr(batch, 'source_code')}")
        
        # Test attention mapping
        dummy_attention = torch.rand(batch.num_nodes)
        
        # Map attention to lines for each graph in batch
        for graph_idx in range(batch.num_graphs):
            graph_mask = (batch.batch == graph_idx)
            graph_attention = dummy_attention[graph_mask]
            graph_line_numbers = batch.line_numbers[graph_mask]
            
            line_attention = LineNumberTracker.map_attention_to_lines(
                graph_attention, graph_line_numbers
            )
            aggregated = LineNumberTracker.aggregate_line_attention(line_attention)
            
            print(f"  Graph {graph_idx} line attention: {aggregated}")
        
        if batch_idx >= 1:  # Only test first 2 batches
            break
    
    # Test individual sample access
    sample_info = dataset.get_sample_with_lines(0)
    print(f"\n📄 Sample 0 details:")
    print(f"  Function: {sample_info['function_name']}")
    print(f"  Label: {sample_info['label']}")
    print(f"  Lines with attention:")
    for line_info in sample_info['line_info'][:3]:  # First 3 lines
        print(f"    Line {line_info['line_number']}: {line_info['attention_score']:.3f} - {line_info['line_content']}")
    
    print(f"\n✅ Enhanced data manager test completed!")


if __name__ == "__main__":
    test_enhanced_data_manager()