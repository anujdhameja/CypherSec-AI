#!/usr/bin/env python3
"""
Use the REAL Config 9 model that achieved 83.57% accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from torch_geometric.loader import DataLoader
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.objects.input_dataset import InputDataset


class RealConfig9Model(nn.Module):
    """
    The REAL Config 9 model that achieved 83.57% test accuracy
    - hidden_dim: 384
    - num_steps: 4  
    - pooling: attention
    """
    
    def __init__(self, input_dim=100, output_dim=2, hidden_dim=384, 
                 num_steps=4, dropout=0.2):
        super().__init__()
        
        print(f"\n=== REAL Config 9 Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        print(f"✓ EXACT Config 9 parameters")
        print(f"   - GNN steps: {num_steps}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Pooling: attention")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # GNN
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Attention pooling
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        print("="*50 + "\n")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GNN with residual
        x_skip = x
        x = self.ggc(x, edge_index)
        x = self.gnn_bn(x)
        x = F.relu(x + x_skip)
        x = self.dropout(x)
        
        # Attention pooling
        batch_size = batch.max().item() + 1
        x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                node_features = x[mask]
                attn_weights = torch.softmax(self.attention(node_features), dim=0)
                x_pooled[i] = torch.sum(attn_weights * node_features, dim=0)
        
        x = x_pooled
        
        # Classifier
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


def test_real_config9():
    """Test the real Config 9 model"""
    
    print("="*80)
    print("TESTING REAL CONFIG 9 MODEL")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with EXACT Config 9 parameters
    model = RealConfig9Model(
        input_dim=100,
        output_dim=2,
        hidden_dim=384,  # REAL Config 9
        num_steps=4,     # REAL Config 9
        dropout=0.2
    )
    
    # Load the REAL Config 9 weights
    print(f"📦 Loading REAL Config 9 weights...")
    state_dict = torch.load('best_optimized_model.pth', map_location=device)
    model.load_state_dict(state_dict, strict=True)
    print(f"✅ REAL Config 9 weights loaded successfully!")
    
    model = model.to(device)
    model.eval()
    
    # Load test data
    print(f"\n📊 Loading test dataset...")
    dataset = InputDataset('data/input')
    
    # Use same split as optimization
    from sklearn.model_selection import train_test_split
    indices = list(range(len(dataset)))
    labels = [int(dataset[i].y.item()) for i in indices]
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, 
        stratify=[labels[i] for i in temp_idx]
    )
    
    test_dataset = [dataset[i] for i in test_idx]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Test the model
    print(f"\n🧪 Testing REAL Config 9 model...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            pred = output.argmax(dim=1)
            correct += (pred == batch.y.long()).sum().item()
            total += batch.y.size(0)
    
    test_acc = correct / total
    
    print(f"\n🎯 REAL CONFIG 9 RESULTS:")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Expected: 83.57%")
    print(f"  Difference: {test_acc - 0.8357:+.4f}")
    
    if abs(test_acc - 0.8357) < 0.02:
        print(f"\n✅ SUCCESS! Model matches expected performance!")
        print(f"🚀 This is the CORRECT Config 9 model to use!")
    else:
        print(f"\n⚠️  Performance differs from expected.")
    
    return model, test_acc


def update_configs_for_real_config9():
    """Update configs.json to use REAL Config 9 parameters"""
    
    print(f"\n🔧 Updating configs.json for REAL Config 9...")
    
    import json
    
    # Read current config
    with open('configs.json', 'r') as f:
        config = json.load(f)
    
    # Update with REAL Config 9 parameters
    config['devign']['model']['gated_graph_conv_args']['out_channels'] = 384
    config['devign']['model']['gated_graph_conv_args']['num_layers'] = 4
    
    # Save updated config
    with open('configs.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated configs.json:")
    print(f"   - hidden_dim: 256 → 384")
    print(f"   - num_steps: 5 → 4")
    
    # Copy the real model to the expected location
    import shutil
    shutil.copy('best_optimized_model.pth', 'models/production_model_config9_v1.0.pth')
    print(f"✅ Copied real Config 9 model to models/production_model_config9_v1.0.pth")


if __name__ == '__main__':
    # Test the real Config 9 model
    model, test_acc = test_real_config9()
    
    if test_acc > 0.80:
        print(f"\n" + "="*80)
        print("READY TO USE REAL CONFIG 9!")
        print("="*80)
        
        # Update configs
        update_configs_for_real_config9()
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Stop current training (Ctrl+C)")
        print(f"   2. Run: python main.py -p")
        print(f"   3. Should achieve ~83.57% test accuracy!")
        
    else:
        print(f"\n❌ Model not performing as expected. Check implementation.")