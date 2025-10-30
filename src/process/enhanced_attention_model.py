#!/usr/bin/env python3
"""
Enhanced Attention Model for Line-Level Vulnerability Detection

This model integrates with our existing Devign system and provides:
- Multi-head attention for diverse pattern detection
- Vulnerability-aware attention focusing on dangerous code patterns
- Line-level vulnerability detection with attention scores
- Compatible with our existing model weights and architecture

Compatible with:
- Our existing BalancedDevignModel (input_dim=100, hidden_dim=256, num_steps=5)
- Current model weights: models/final_model_with_attention.pth
- Existing CLI interfaces and prediction systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool


class VulnerabilityAwareAttention(nn.Module):
    """
    Multi-head attention mechanism that learns to focus on vulnerable code patterns.
    Uses both node features and graph context to identify suspicious lines.
    
    Key improvements over simple attention:
    - Multi-head attention for diverse pattern detection
    - Vulnerability-specific pattern detector
    - Context-aware attention refinement
    - Temperature scaling for peaked distributions
    """
    
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        print(f"✓ VulnerabilityAwareAttention: {hidden_dim}D with {num_heads} heads")
        
        # Multi-head attention components
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Vulnerability-specific pattern detector (PRIMARY SIGNAL)
        # This learns to detect dangerous patterns like strcpy, printf, etc.
        self.vulnerability_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output [0, 1] for vulnerability probability
        )
        
        # Context-aware attention refinement
        # Considers both node features and graph-level context
        self.context_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Node + Graph context
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        print(f"✓ Vulnerability detector: {hidden_dim} → {hidden_dim//2} → 1")
        print(f"✓ Context attention: {hidden_dim*2} → {hidden_dim} → 1")
    
    def forward(self, x, batch, return_attention=True):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]
            return_attention: Whether to return attention weights
            
        Returns:
            attended_features: [num_nodes, hidden_dim]
            attention_weights: [num_nodes] (if return_attention=True)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        batch_size = batch.max().item() + 1
        num_nodes = x.size(0)
        
        # Multi-head self-attention
        Q = self.query_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores per head
        attention_scores = torch.einsum('nhd,mhd->nmh', Q, K) / (self.head_dim ** 0.5)
        
        # Mask attention to same graph only (prevent cross-graph attention)
        mask = (batch.unsqueeze(1) == batch.unsqueeze(0)).float()
        mask = mask.unsqueeze(-1).expand_as(attention_scores)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax per head
        attention_probs = F.softmax(attention_scores, dim=1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        attended = torch.einsum('nmh,mhd->nhd', attention_probs, V)
        attended = attended.reshape(num_nodes, self.hidden_dim)
        
        # === VULNERABILITY PATTERN DETECTION (PRIMARY SIGNAL) ===
        # This is the key improvement: focus on vulnerability patterns
        vulnerability_scores = self.vulnerability_detector(x).squeeze(-1)
        
        # === CONTEXT-AWARE ATTENTION (SECONDARY SIGNAL) ===
        # Consider graph-level context for each node
        graph_context = global_max_pool(x, batch)[batch]  # Broadcast graph representation
        combined = torch.cat([x, graph_context], dim=1)
        context_scores = self.context_attention(combined).squeeze(-1)
        
        # === COMBINE ATTENTION SIGNALS ===
        # Weighted combination with vulnerability detection as primary signal
        multi_head_avg = attention_probs.mean(dim=(1, 2))  # Average across heads and positions
        
        final_weights = (
            0.6 * vulnerability_scores +      # PRIMARY: Vulnerability patterns (increased weight)
            0.3 * context_scores +            # SECONDARY: Graph context  
            0.1 * multi_head_avg              # TERTIARY: Multi-head attention
        )
        
        # === NORMALIZE WEIGHTS PER GRAPH ===
        # Apply softmax normalization within each graph for interpretability
        attention_per_graph = []
        for i in range(batch_size):
            graph_mask = (batch == i)
            if graph_mask.sum() > 0:
                graph_weights = final_weights[graph_mask]
                # Temperature scaling for peaked distributions (better line-level detection)
                normalized = F.softmax(graph_weights * 5.0, dim=0)  # Temperature = 5.0
                attention_per_graph.append(normalized)
            else:
                attention_per_graph.append(torch.tensor([]))
        
        # Reconstruct full attention tensor
        final_attention = torch.zeros_like(final_weights)
        for i in range(batch_size):
            graph_mask = (batch == i)
            if graph_mask.sum() > 0:
                final_attention[graph_mask] = attention_per_graph[i]
        
        # === APPLY ATTENTION TO FEATURES ===
        attended_features = attended * final_attention.unsqueeze(-1)
        attended_features = self.layer_norm(attended_features + x)  # Residual connection
        
        if return_attention:
            return attended_features, final_attention
        return attended_features


class EnhancedAttentionDevignModel(nn.Module):
    """
    Enhanced Devign model with vulnerability-aware attention for line-level detection.
    
    Compatible with our existing system:
    - Same input/output dimensions as BalancedDevignModel
    - Can load weights from existing models
    - Integrates with current CLI interfaces
    """
    
    def __init__(self, input_dim=100, output_dim=2, hidden_dim=256, 
                 num_steps=5, num_attention_heads=4, dropout=0.2, pooling='mean_max'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.pooling = pooling
        
        print(f"\n=== Enhanced Attention Devign Model ===")
        print(f"Input dim: {input_dim}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Output dim: {output_dim}")
        print(f"GNN steps: {num_steps}")
        print(f"Attention heads: {num_attention_heads}")
        print(f"Pooling: {pooling}")
        print(f"Dropout: {dropout}")
        
        # Input projection (compatible with existing models)
        if input_dim != hidden_dim:
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            print(f"✓ Input projection: {input_dim} → {hidden_dim}")
        else:
            self.input_projection = None
            print(f"✓ No projection needed")
        
        # Gated Graph Neural Network (same as existing model)
        self.ggnn = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=num_steps,
            aggr='add'
        )
        print(f"✓ GatedGraphConv: {hidden_dim} channels, {num_steps} steps")
        
        # Enhanced multi-head attention (KEY IMPROVEMENT)
        self.attention = VulnerabilityAwareAttention(
            hidden_dim=hidden_dim, 
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Classifier (compatible with existing architecture)
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
        Forward pass with optional attention weights for explainability.
        
        Args:
            data: PyG Data object with x, edge_index, batch
            return_attention: Return attention weights for line-level analysis
            
        Returns:
            logits: [batch_size, output_dim]
            attention_weights: [num_nodes] (if return_attention=True)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Step 1: Project input features if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Step 2: Message passing to learn graph structure
        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        
        # Step 3: Apply vulnerability-aware attention (KEY IMPROVEMENT)
        if return_attention:
            x_attended, attention_weights = self.attention(x, batch, return_attention=True)
        else:
            x_attended = self.attention(x, batch, return_attention=False)
            attention_weights = None
        
        # Step 4: Graph-level pooling
        if self.pooling == 'mean':
            graph_embedding = global_mean_pool(x_attended, batch)
        elif self.pooling == 'max':
            graph_embedding = global_max_pool(x_attended, batch)
        elif self.pooling == 'mean_max':
            mean_pool = global_mean_pool(x_attended, batch)
            max_pool = global_max_pool(x_attended, batch)
            graph_embedding = torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Step 5: Classification
        graph_embedding = self.dropout_layer(graph_embedding)
        logits = self.classifier(graph_embedding)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_line_level_vulnerabilities(self, data, node_to_line_mapping=None, threshold=0.1, top_k=10):
        """
        Extract line-level vulnerability information using attention weights.
        
        Args:
            data: PyG Data object
            node_to_line_mapping: Dict[node_idx -> line_number] (optional)
            threshold: Attention threshold for considering a line vulnerable
            top_k: Maximum number of lines to return
            
        Returns:
            vulnerable_lines: List of vulnerability information per line
        """
        self.eval()
        with torch.no_grad():
            logits, attention_weights = self.forward(data, return_attention=True)
            
            # Get prediction details
            probs = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
            
            # Sort nodes by attention weight
            attention_np = attention_weights.cpu().numpy()
            sorted_indices = np.argsort(attention_np)[::-1]  # Descending order
            
            vulnerable_lines = []
            max_attention = attention_np.max()
            
            for i, node_idx in enumerate(sorted_indices[:top_k]):
                score = attention_np[node_idx]
                
                # Only consider nodes above threshold
                if score < threshold:
                    break
                
                # Map node to line number
                if node_to_line_mapping and node_idx in node_to_line_mapping:
                    line_num = node_to_line_mapping[node_idx]
                else:
                    line_num = node_idx  # Use node index as fallback
                
                # Determine risk level based on attention score
                if score > 0.15:  # Top 10-20% typically
                    risk_level = "HIGH"
                elif score > 0.08:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
                
                vulnerable_lines.append({
                    'line_number': line_num,
                    'node_index': node_idx,
                    'attention_score': float(score),
                    'risk_level': risk_level,
                    'relative_importance': float(score / max_attention),
                    'rank': i + 1
                })
            
            return {
                'prediction': {
                    'class': pred_class,
                    'confidence': confidence,
                    'is_vulnerable': pred_class == 1
                },
                'vulnerable_lines': vulnerable_lines,
                'attention_stats': {
                    'mean': float(np.mean(attention_np)),
                    'std': float(np.std(attention_np)),
                    'max': float(np.max(attention_np)),
                    'min': float(np.min(attention_np)),
                    'num_high_attention': int(np.sum(attention_np > 0.15)),
                    'num_medium_attention': int(np.sum((attention_np > 0.08) & (attention_np <= 0.15))),
                    'num_low_attention': int(np.sum(attention_np <= 0.08))
                }
            }


class AttentionSupervisionLoss(nn.Module):
    """
    Custom loss that supervises attention to focus on vulnerable lines.
    Uses weak supervision from vulnerability annotations.
    
    This helps train the model to pay attention to the RIGHT lines.
    """
    
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for attention supervision loss
        self.ce_loss = nn.CrossEntropyLoss()
        
        print(f"✓ AttentionSupervisionLoss: α={alpha} (classification), β={beta} (attention)")
    
    def forward(self, logits, labels, attention_weights, vulnerable_line_masks=None):
        """
        Args:
            logits: [batch_size, num_classes]
            labels: [batch_size] - 0=safe, 1=vulnerable
            attention_weights: [num_nodes]
            vulnerable_line_masks: [num_nodes] - 1.0 for vulnerable lines, 0.0 otherwise
            
        Returns:
            total_loss: Combined classification + attention supervision loss
            cls_loss: Classification loss component
            attn_loss: Attention supervision loss component
        """
        # Classification loss (primary objective)
        cls_loss = self.ce_loss(logits, labels)
        
        # Attention supervision loss (secondary objective)
        if vulnerable_line_masks is not None and vulnerable_line_masks.sum() > 0:
            # Encourage high attention on vulnerable lines, low on others
            vulnerable_loss = F.binary_cross_entropy(
                attention_weights, 
                vulnerable_line_masks, 
                reduction='mean'
            )
            
            # Encourage attention diversity (avoid uniform distribution)
            # This prevents the model from giving equal attention to all lines
            entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum()
            diversity_loss = -entropy / attention_weights.size(0)  # Negative entropy
            
            attn_loss = vulnerable_loss + 0.1 * diversity_loss
        else:
            attn_loss = torch.tensor(0.0, device=logits.device)
        
        # Combined loss
        total_loss = self.alpha * cls_loss + self.beta * attn_loss
        
        return total_loss, cls_loss, attn_loss


def convert_existing_model_to_enhanced(original_model_path, enhanced_model_path):
    """
    Convert existing AttentionDevignModel to EnhancedAttentionDevignModel.
    
    This allows us to upgrade our current model to the enhanced version
    while preserving as much training as possible.
    """
    print(f"\n🔄 Converting model to enhanced attention version...")
    print(f"Source: {original_model_path}")
    print(f"Target: {enhanced_model_path}")
    
    # Load original model weights
    original_state_dict = torch.load(original_model_path, map_location='cpu')
    
    # Create enhanced model with same architecture
    enhanced_model = EnhancedAttentionDevignModel(
        input_dim=100,
        output_dim=2,
        hidden_dim=256,
        num_steps=5,
        num_attention_heads=4,
        dropout=0.2,
        pooling='mean_max'
    )
    
    # Transfer compatible weights
    enhanced_state_dict = enhanced_model.state_dict()
    transferred_keys = []
    
    # Map old keys to new keys
    key_mapping = {
        'input_projection.0.weight': 'input_projection.0.weight',
        'input_projection.0.bias': 'input_projection.0.bias',
        'ggc.weight': 'ggnn.weight',
        'ggc.rnn.weight_ih': 'ggnn.rnn.weight_ih',
        'ggc.rnn.weight_hh': 'ggnn.rnn.weight_hh',
        'ggc.rnn.bias_ih': 'ggnn.rnn.bias_ih',
        'ggc.rnn.bias_hh': 'ggnn.rnn.bias_hh',
        'classifier.0.weight': 'classifier.0.weight',
        'classifier.0.bias': 'classifier.0.bias',
        'classifier.2.weight': 'classifier.2.weight',
        'classifier.2.bias': 'classifier.2.bias',
        'classifier.4.weight': 'classifier.4.weight',
        'classifier.4.bias': 'classifier.4.bias',
    }
    
    for old_key, new_key in key_mapping.items():
        if old_key in original_state_dict and new_key in enhanced_state_dict:
            if original_state_dict[old_key].shape == enhanced_state_dict[new_key].shape:
                enhanced_state_dict[new_key] = original_state_dict[old_key]
                transferred_keys.append(f"{old_key} → {new_key}")
    
    # Load transferred weights
    enhanced_model.load_state_dict(enhanced_state_dict)
    
    print(f"✅ Transferred {len(transferred_keys)} weight tensors")
    for key in transferred_keys:
        print(f"   ✓ {key}")
    
    # Save enhanced model
    torch.save(enhanced_model.state_dict(), enhanced_model_path)
    print(f"💾 Enhanced model saved to: {enhanced_model_path}")
    
    return enhanced_model


# ============================================================================
# TESTING AND DEMO FUNCTIONS
# ============================================================================

def test_enhanced_attention_model():
    """Test the enhanced attention model with dummy data"""
    print("\n🧪 Testing Enhanced Attention Model...")
    
    # Create model
    model = EnhancedAttentionDevignModel(
        input_dim=100,
        output_dim=2,
        hidden_dim=256,
        num_steps=5,
        num_attention_heads=4,
        dropout=0.2
    )
    
    # Create dummy data
    from torch_geometric.data import Data
    
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
        logits, attention_weights = model(data, return_attention=True)
        
        print(f"✅ Forward pass successful!")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        print(f"   Attention range: [{attention_weights.min():.3f}, {attention_weights.max():.3f}]")
        
        # Test line-level vulnerability detection
        vulnerabilities = model.get_line_level_vulnerabilities(data, top_k=5)
        
        print(f"\n📊 Line-Level Vulnerability Analysis:")
        print(f"   Prediction: {'Vulnerable' if vulnerabilities['prediction']['is_vulnerable'] else 'Safe'}")
        print(f"   Confidence: {vulnerabilities['prediction']['confidence']:.2%}")
        
        print(f"\n🎯 Top 5 Most Suspicious Lines:")
        for vuln in vulnerabilities['vulnerable_lines']:
            print(f"   #{vuln['rank']}: Line {vuln['line_number']} "
                  f"(attention: {vuln['attention_score']:.3f}, risk: {vuln['risk_level']})")
        
        print(f"\n📈 Attention Statistics:")
        stats = vulnerabilities['attention_stats']
        print(f"   Mean: {stats['mean']:.3f}")
        print(f"   Max: {stats['max']:.3f}")
        print(f"   High attention lines (>0.15): {stats['num_high_attention']}")
        print(f"   Medium attention lines (0.08-0.15): {stats['num_medium_attention']}")
        print(f"   Low attention lines (<0.08): {stats['num_low_attention']}")


if __name__ == "__main__":
    # Test the enhanced attention model
    test_enhanced_attention_model()
    
    # Example: Convert existing model to enhanced version
    # convert_existing_model_to_enhanced(
    #     'models/final_model_with_attention.pth',
    #     'models/enhanced_attention_model.pth'
    # )