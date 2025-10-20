"""
NUCLEAR OPTION: Simplest Possible Working Baseline
This WILL work - if this doesn't work, there's something wrong with the data

This bypasses all your existing code and trains a simple model from scratch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import pandas as pd
from pathlib import Path
import time


# ============================================
# SIMPLEST POSSIBLE MODEL
# ============================================

class SimplestGNN(nn.Module):
    """
    Absolute simplest GNN that should work
    No fancy features, just basic graph convolution
    """
    def __init__(self, input_dim=205, hidden_dim=128, output_dim=2):
        super().__init__()
        
        # Two GCN layers (standard, well-tested)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Simple classifier
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        
        print(f"\n✓ SimplestGNN created: {input_dim} → {hidden_dim} → {output_dim}")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x


# ============================================
# TRAINING FUNCTION
# ============================================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Simple, clean training loop"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        # Forward
        optimizer.zero_grad()
        output = model(batch)
        target = batch.y.squeeze().long()
        
        # Loss
        loss = criterion(output, target)
        
        # Check for issues
        if torch.isnan(loss):
            print(f"⚠️ NaN loss at batch {batch_idx}")
            continue
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update
        optimizer.step()
        
        # Stats
        total_loss += loss.item() * batch.num_graphs
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += batch.num_graphs
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Simple validation loop"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            output = model(batch)
            target = batch.y.squeeze().long()
            
            loss = criterion(output, target)
            
            total_loss += loss.item() * batch.num_graphs
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += batch.num_graphs
            
            all_preds.extend(pred.cpu().tolist())
            all_targets.extend(target.cpu().tolist())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Check if stuck
    unique_preds = len(set(all_preds))
    if unique_preds == 1:
        print(f"    ⚠️ WARNING: Predicting only class {all_preds[0]}")
    
    return avg_loss, accuracy


# ============================================
# MAIN SCRIPT
# ============================================

def main():
    print("\n" + "="*80)
    print("SIMPLE BASELINE TEST")
    print("="*80)
    print("\nThis is the simplest possible GNN that should work.")
    print("If this doesn't work, there's an issue with the data itself.\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    
    # Find all input files
    input_path = Path('data/input')
    input_files = list(input_path.glob('*_input.pkl'))
    
    if not input_files:
        print("❌ No input files found in data/input/")
        return
    
    print(f"   Found {len(input_files)} input files")
    
    # Load and combine all data
    all_data = []
    for f in input_files:
        df = pd.read_pickle(f)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"   Total samples: {len(combined)}")
    
    # Extract graphs
    graphs = combined['input'].tolist()
    
    # Split data
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size+val_size]
    test_graphs = graphs[train_size+val_size:]
    
    print(f"   Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Create loaders
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    # Create model
    print("\n2. Creating model...")
    # Update input_dim to match our new embedding dimension of 100
    model = SimplestGNN(input_dim=100, hidden_dim=128, output_dim=2).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"   Optimizer: Adam(lr=0.001)")
    print(f"   Loss: CrossEntropyLoss")
    
    # Training
    print("\n3. Training...")
    print("-" * 80)
    
    num_epochs = 20
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Debug first epoch
        if epoch == 0:  # First epoch only
            print("\n=== DEBUGGING FIRST EPOCH ===")
            
            # Check validation predictions
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    pred = output.argmax(dim=1)
                    target = batch.y.squeeze().long()
                    
                    val_preds.extend(pred.cpu().tolist())
                    val_targets.extend(target.cpu().tolist())
            
            from collections import Counter
            print(f"Predictions: {Counter(val_preds)}")
            print(f"Targets: {Counter(val_targets)}")
            print(f"Unique predictions: {set(val_preds)}")
            print("="*50)
        
        # Time
        elapsed = time.time() - start_time
        
        # Print
        print(f"Epoch {epoch+1:2d}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2%} | "
              f"Time={elapsed:.1f}s")
        
        # Check for issues
        if epoch > 5 and val_acc < 0.52:
            print(f"    ⚠️ Val acc still low after {epoch+1} epochs")
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model
            torch.save(model.state_dict(), 'simple_baseline_best.pth')
    
    print("-" * 80)
    print(f"\n✓ Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2%}")
    
    # Analysis
    print("\n" + "="*80)
    print("RESULT ANALYSIS")
    print("="*80)
    
    if best_val_acc > 0.55:
        print(f"\n✓ SUCCESS! Model achieved {best_val_acc:.2%}")
        print("\nThis proves:")
        print("  1. Your DATA is fine")
        print("  2. The problem was in the complex model/training setup")
        print("\nNext steps:")
        print("  - Your data pipeline is working correctly")
        print("  - The issue is in your Devign model configuration")
        print("  - Check your model's forward pass and loss calculation")
        
    elif best_val_acc > 0.52:
        print(f"\n⚠️ MARGINAL: Model achieved {best_val_acc:.2%}")
        print("\nThis is slightly better than random (50%)")
        print("Possible issues:")
        print("  - Features might not be very discriminative")
        print("  - Need a more complex model")
        print("  - Or more training epochs")
        
    else:
        print(f"\n❌ FAIL: Model only achieved {best_val_acc:.2%}")
        print("\nThis suggests a DATA problem:")
        print("  - Check if features (Word2Vec embeddings) are meaningful")
        print("  - Check if targets are correct")
        print("  - Run the diagnostic script again on this simple model")


if __name__ == "__main__":
    main()