#!/usr/bin/env python3
"""
Attention-Enhanced Devign Model for Explainable Vulnerability Detection

This model extends the original BalancedDevignModel with attention mechanism
to identify which nodes (code lines) the classifier pays attention to.

Key Features:
- Returns attention weights for each node [0.0 - 1.0]
- 0.9+ = Very important for vulnerability detection
- 0.5 = Somewhat important  
- 0.1 = Not important
- 0.0 = Ignored

Usage:
    model = AttentionDevignModel(input_dim=100, output_dim=2, hidden_dim=256)
    output, attention_weights = model(data)
    # attention_weights shape: [num_nodes] with values 0.0-1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool


class AttentionDevignModel(nn.Module):
    """
    Attention-Enhanced Devign Model
    
    Architecture:
    1. Input Projection: input_dim → hidden_dim
    2. GatedGraphConv layers (message passing)
    3. Attention mechanism: computes importance weights per node
    4. Dual pooling: mean + max pooling with attention weights
    5. Classification head
    
    Returns:
        - predictions: [batch_size, output_dim]
        - attention_weights: [num_nodes] values in [0.0, 1.0]
    """
    
    def __init__(self, input_dim=100, output_dim=2, hidden_dim=256, 
                 num_steps=5, dropout=0.2, pooling='mean_max'):
        """
        Args:
            input_dim: Node feature dimension (e.g., 100 for Word2Vec)
            output_dim: Number of classes (2 for binary classification)
            hidden_dim: Hidden dimension for GNN layers
            num_steps: Number of GatedGraphConv message passing steps
            dropout: Dropout probability
            pooling: Pooling strategy ('mean', 'max', 'mean_max')
        """
        super(AttentionDevignModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dropout = dropout
        self.pooling = pooling
        
        print(f"\n=== Attention-Enhanced Devign Model ===")
        print(f"Input dim: {input_dim}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Output dim: {output_dim}")
        print(f"GNN steps: {num_steps}")
        print(f"Pooling: {pooling}")
        print(f"Dropout: {dropout}")
        
        # Input projection layer
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            print(f"✓ Input projection: {input_dim} → {hidden_dim}")
        else:
            self.input_projection = None
            print(f"✓ No projection needed")
        
        # GatedGraphConv layer for message passing
        self.ggc = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=num_steps,
            aggr='add'
        )
        print(f"✓ GatedGraphConv: {hidden_dim} channels, {num_steps} steps")
        
        # ===== ATTENTION MECHANISM =====
        # This is the key addition: attention layer to compute node importance
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Output single attention score per node
            nn.Sigmoid()  # Ensure attention weights are in [0, 1]
        )
        print(f"✓ Attention layer: {hidden_dim} → {hidden_dim//2} → 1")
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Classification head
        # Input size depends on pooling strategy
        if pooling == 'mean_max':
            classifier_input_dim = hidden_dim * 2  # Concatenated mean + max
        else:
            classifier_input_dim = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        print(f"✓ Classifier: {classifier_input_dim} → {hidden_dim} → {hidden_dim//2} → {output_dim}")
        print(f"{'='*50}\n")
    
    def forward(self, data, return_attention=True):
        """
        Forward pass with attention mechanism
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment vector [num_nodes]
            return_attention: Whether to return attention weights
        
        Returns:
            if return_attention:
                (predictions, attention_weights)
            else:
                predictions
            
            predictions: [batch_size, output_dim]
            attention_weights: [num_nodes] with values in [0.0, 1.0]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Step 1: Project input features if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
            x = F.relu(x)
        
        # Step 2: GatedGraphConv message passing
        # This updates node representations based on graph structure
        x = self.ggc(x, edge_index)
        x = F.relu(x)
        
        # Step 3: ATTENTION MECHANISM
        # Compute attention weights for each node
        attention_weights = self.attention_layer(x)  # [num_nodes, 1]
        attention_weights = attention_weights.squeeze(-1)  # [num_nodes]
        
        # Step 4: Attention-weighted pooling
        # Apply attention weights to node features before pooling
        weighted_x = x * attention_weights.unsqueeze(-1)  # [num_nodes, hidden_dim]
        
        if self.pooling == 'mean':
            # Weighted mean pooling
            graph_repr = global_mean_pool(weighted_x, batch)
            
        elif self.pooling == 'max':
            # Weighted max pooling
            graph_repr = global_max_pool(weighted_x, batch)
            
        elif self.pooling == 'mean_max':
            # Dual pooling: concatenate weighted mean and max
            mean_pool = global_mean_pool(weighted_x, batch)
            max_pool = global_max_pool(weighted_x, batch)
            graph_repr = torch.cat([mean_pool, max_pool], dim=1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Step 5: Classification
        graph_repr = self.dropout_layer(graph_repr)
        predictions = self.classifier(graph_repr)
        
        if return_attention:
            return predictions, attention_weights
        else:
            return predictions
    
    def get_attention_explanation(self, data, node_labels=None, top_k=5):
        """
        Get human-readable explanation of attention weights
        
        Args:
            data: PyTorch Geometric Data object
            node_labels: Optional list of node labels/descriptions
            top_k: Number of top attention nodes to return
        
        Returns:
            dict: {
                'top_nodes': List of (node_id, attention_weight, label) tuples
                'attention_stats': Statistics about attention distribution
                'risk_assessment': Risk level based on attention pattern
            }
        """
        self.eval()
        with torch.no_grad():
            predictions, attention_weights = self.forward(data, return_attention=True)
            
            # Get prediction details
            probs = torch.softmax(predictions, dim=1)
            pred_class = predictions.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
            
            # Sort nodes by attention weight
            attention_np = attention_weights.cpu().numpy()
            sorted_indices = np.argsort(attention_np)[::-1]  # Descending order
            
            # Get top-k nodes
            top_nodes = []
            for i in range(min(top_k, len(sorted_indices))):
                node_id = sorted_indices[i]
                attention_score = attention_np[node_id]
                
                if node_labels and node_id < len(node_labels):
                    label = node_labels[node_id]
                else:
                    label = f"Node_{node_id}"
                
                top_nodes.append((node_id, attention_score, label))
            
            # Attention statistics
            attention_stats = {
                'mean': float(np.mean(attention_np)),
                'std': float(np.std(attention_np)),
                'max': float(np.max(attention_np)),
                'min': float(np.min(attention_np)),
                'num_high_attention': int(np.sum(attention_np > 0.7)),
                'num_medium_attention': int(np.sum((attention_np > 0.3) & (attention_np <= 0.7))),
                'num_low_attention': int(np.sum(attention_np <= 0.3))
            }
            
            # Risk assessment based on attention pattern
            max_attention = attention_stats['max']
            high_attention_nodes = attention_stats['num_high_attention']
            
            if pred_class == 1:  # Vulnerable
                if max_attention > 0.8 and high_attention_nodes >= 2:
                    risk_level = "Critical"
                elif max_attention > 0.6 and high_attention_nodes >= 1:
                    risk_level = "High"
                elif max_attention > 0.4:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
            else:  # Safe
                risk_level = "Low"
            
            return {
                'prediction': {
                    'class': pred_class,
                    'confidence': confidence,
                    'is_vulnerable': pred_class == 1
                },
                'top_nodes': top_nodes,
                'attention_stats': attention_stats,
                'risk_assessment': risk_level,
                'all_attention_weights': attention_np.tolist()
            }


def convert_model_to_attention(original_model_path, attention_model_path):
    """
    Convert a trained BalancedDevignModel to AttentionDevignModel
    
    This function loads weights from the original model and adapts them
    for the attention-enhanced version.
    
    Args:
        original_model_path: Path to trained BalancedDevignModel
        attention_model_path: Path to save AttentionDevignModel
    """
    print(f"\n🔄 Converting model to attention-enhanced version...")
    print(f"Source: {original_model_path}")
    print(f"Target: {attention_model_path}")
    
    # Load original model weights
    original_state_dict = torch.load(original_model_path, map_location='cpu')
    
    # Create attention model with same architecture
    attention_model = AttentionDevignModel(
        input_dim=100,
        output_dim=2,
        hidden_dim=256,
        num_steps=5,
        dropout=0.2,
        pooling='mean_max'
    )
    
    # Transfer compatible weights
    attention_state_dict = attention_model.state_dict()
    transferred_keys = []
    
    for key in original_state_dict:
        if key in attention_state_dict:
            if original_state_dict[key].shape == attention_state_dict[key].shape:
                attention_state_dict[key] = original_state_dict[key]
                transferred_keys.append(key)
    
    # Load transferred weights
    attention_model.load_state_dict(attention_state_dict)
    
    print(f"✅ Transferred {len(transferred_keys)} weight tensors")
    print(f"📝 Transferred keys: {transferred_keys}")
    
    # Save attention model
    torch.save(attention_model.state_dict(), attention_model_path)
    print(f"💾 Attention model saved to: {attention_model_path}")
    
    return attention_model


# ============================================
# Testing and Demo Functions
# ============================================

def test_attention_model():
    """Test the attention model with dummy data"""
    print("\n🧪 Testing Attention Model...")
    
    # Create model
    model = AttentionDevignModel(
        input_dim=100,
        output_dim=2,
        hidden_dim=256,
        num_steps=5,
        dropout=0.2
    )
    
    # Create dummy data
    from torch_geometric.data import Data
    import numpy as np
    
    num_nodes = 15
    x = torch.randn(num_nodes, 100)
    
    # Create edges (simple chain + some connections)
    edges = []
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    
    # Add some complex connections
    edges.extend([[0, 5], [5, 0], [2, 8], [8, 2], [3, 12], [12, 3]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model(data, return_attention=True)
        
        print(f"✅ Forward pass successful!")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        print(f"   Attention range: [{attention_weights.min():.3f}, {attention_weights.max():.3f}]")
        
        # Test explanation
        explanation = model.get_attention_explanation(data, top_k=5)
        
        print(f"\n📊 Attention Analysis:")
        print(f"   Prediction: {'Vulnerable' if explanation['prediction']['is_vulnerable'] else 'Safe'}")
        print(f"   Confidence: {explanation['prediction']['confidence']:.2%}")
        print(f"   Risk Level: {explanation['risk_assessment']}")
        
        print(f"\n🎯 Top 5 Most Important Nodes:")
        for i, (node_id, attention, label) in enumerate(explanation['top_nodes']):
            print(f"   #{i+1}: {label} (attention: {attention:.3f})")
        
        print(f"\n📈 Attention Statistics:")
        stats = explanation['attention_stats']
        print(f"   Mean: {stats['mean']:.3f}")
        print(f"   Max: {stats['max']:.3f}")
        print(f"   High attention nodes (>0.7): {stats['num_high_attention']}")
        print(f"   Medium attention nodes (0.3-0.7): {stats['num_medium_attention']}")
        print(f"   Low attention nodes (<0.3): {stats['num_low_attention']}")


if __name__ == "__main__":
    # Test the attention model
    test_attention_model()
    
    # Example: Convert existing model to attention version
    # convert_model_to_attention(
    #     'models/final_model.pth',
    #     'models/final_model_with_attention.pth'
    # )