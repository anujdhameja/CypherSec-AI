"""
Stable Training Configuration
Fixes the gradient instability and erratic behavior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_max_pool


class StableDevignModel(nn.Module):
    """
    STABLE version of Devign with:
    1. Reduced GNN steps (8 → 4) to prevent gradient explosion
    2. Batch normalization after every layer
    3. Gradient clipping built-in
    4. Residual connections
    """
    
    def __init__(self, input_dim=205, output_dim=2, hidden_dim=200, num_steps=4, dropout=0.3):  # REDUCED from 8 to 4 steps
        super().__init__()
        
        print(f"\n=== STABLE Devign Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        print(f"⚠️ Using {num_steps} GNN steps (reduced from 8 for stability)")
        
        # Input projection + BN
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # GNN with REDUCED steps
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Simpler classifier (removed Conv1d which adds instability)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✓ BatchNorm after each layer")
        print(f"✓ Simplified architecture (no Conv1d)")
        print("="*50 + "\n")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x_proj = self.input_proj(x)
        x_proj = self.input_bn(x_proj)
        x_proj = F.relu(x_proj)
        
        # GNN with residual
        x_gnn = self.ggc(x_proj, edge_index)
        x_gnn = self.gnn_bn(x_gnn)
        
        # Residual connection (helps gradient flow)
        x = F.relu(x_proj + x_gnn)
        
        # Global pooling
        x = global_max_pool(x, batch)
        
        # Classifier
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def save(self, path):
        """Save model"""
        torch.save(self.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.load_state_dict(torch.load(path))
        print(f"✓ Model loaded from {path}")


def train_one_epoch_stable(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Training loop with gradient clipping for stability"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        output = model(batch)
        target = batch.y.squeeze().long()
        
        # Loss
        loss = criterion(output, target)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"⚠️ NaN loss detected at batch {batch_idx}, skipping")
            continue
        
        # Backward
        loss.backward()
        
        # CRITICAL: Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Check for exploding gradients
        if grad_norm > 10.0:
            print(f"⚠️ Large gradient norm: {grad_norm:.2f} at batch {batch_idx}")
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item() * batch.num_graphs
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += batch.num_graphs
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate_stable(model, val_loader, criterion, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            target = batch.y.squeeze().long()
            
            loss = criterion(output, target)
            
            if not torch.isnan(loss):
                total_loss += loss.item() * batch.num_graphs
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += batch.num_graphs
    
    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def stable_training_config():
    """Return recommended stable training configuration"""
    config = {
        # Model parameters
        'input_dim': 205,
        'hidden_dim': 200,
        'output_dim': 2,
        'num_steps': 4,  # REDUCED from 8
        'dropout': 0.3,
        
        # Training parameters
        'learning_rate': 3e-4,  # Slightly higher than 1e-4
        'weight_decay': 1.3e-6,
        'batch_size': 8,
        'epochs': 50,  # Reduced from 100
        'patience': 10,
        
        # Stability parameters
        'gradient_clip': 1.0,
        'use_scheduler': True,
        'scheduler_patience': 5,
        'scheduler_factor': 0.5,
    }
    
    print("\n" + "="*80)
    print("STABLE TRAINING CONFIGURATION")
    print("="*80)
    
    print("\n🔧 Key Stability Fixes:")
    print(f"   - Reduced GNN steps: 8 → {config['num_steps']}")
    print(f"   - Gradient clipping: max_norm={config['gradient_clip']}")
    print(f"   - Learning rate: {config['learning_rate']:.0e} (slightly higher)")
    print(f"   - Epochs: {config['epochs']} (reduced for faster iteration)")
    print(f"   - Batch normalization: After every layer")
    print(f"   - LR scheduler: ReduceLROnPlateau")
    
    print("\n📊 Expected Performance:")
    print("   - Epoch 1-5:   Train ~40-50%, Val ~40-45%")
    print("   - Epoch 10-20: Train ~55-65%, Val ~50-55%")
    print("   - Epoch 30+:   Train ~60-70%, Val ~52-58%")
    
    print("\n⚠️ What to Watch For:")
    print("   - Loss should DECREASE smoothly (no huge jumps)")
    print("   - Train acc should be >= Val acc")
    print("   - Gradient norms should stay < 10")
    
    return config


# ============================================
# Quick Test
# ============================================

def test_stable_model():
    """Test if stable model works"""
    print("\n" + "="*80)
    print("TESTING STABLE MODEL")
    print("="*80)
    
    from torch_geometric.data import Data, Batch
    
    # Create model
    model = StableDevignModel(input_dim=205, hidden_dim=200, num_steps=4)
    
    # Create dummy data
    graphs = []
    for _ in range(4):
        x = torch.randn(10, 205)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long)
        y = torch.randint(0, 2, (1,))
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    
    batch = Batch.from_data_list(graphs)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)
        print(f"\n✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output values:\n{output}")
    
    # Test training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    for i in range(3):
        optimizer.zero_grad()
        output = model(batch)
        target = batch.y.squeeze().long()
        loss = criterion(output, target)
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        
        print(f"\nStep {i+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}, GradNorm={grad_norm:.2f}")
    
    print("\n✓ Training test successful!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Show config
    config = stable_training_config()
    
    # Test model
    test_stable_model()
    
    print("\n🎯 TO USE THIS IN YOUR CODE:")
    print("1. Replace your model with StableDevignModel")
    print("2. Use train_one_epoch_stable() instead of regular training")
    print("3. Add gradient clipping and learning rate scheduler")
    print("4. Reduce num_steps from 8 to 4")