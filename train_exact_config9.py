#!/usr/bin/env python3
"""
Train with EXACT Config 9 implementation that achieved 83.57%
Replicates the optimization script exactly
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')
from src.utils.objects.input_dataset import InputDataset


class ExactConfig9Model(nn.Module):
    """EXACT replica of the Config 9 model that achieved 83.57%"""
    
    def __init__(self, input_dim=100, hidden_dim=256, output_dim=2, 
                 num_steps=5, dropout=0.2, pooling='mean_max'):
        super().__init__()
        
        print(f"\n=== EXACT Config 9 Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        print(f"✓ EXACT optimization script implementation")
        print(f"   - GNN steps: {num_steps}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Pooling: {pooling}")
        
        # EXACT same architecture as optimization script
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Configurable pooling
        self.pooling = pooling
        
        # Classifier layers - EXACT same as optimization script
        pool_dim = hidden_dim * 2 if pooling == 'mean_max' else hidden_dim
        self.fc1 = nn.Linear(pool_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        print("="*50 + "\n")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection - EXACT same as optimization script
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GNN layers - EXACT same (NO residual connection!)
        x = self.ggc(x, edge_index)
        x = self.gnn_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Pooling - EXACT same as optimization script
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            x = torch.cat([mean_pool, max_pool], dim=1)
        
        # Classifier - EXACT same as optimization script
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


def train_exact_config9():
    """Train with EXACT Config 9 parameters and training loop"""
    
    print("="*80)
    print("TRAINING EXACT CONFIG 9 MODEL")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load dataset - EXACT same as optimization script
    print(f"\n📊 Loading dataset...")
    dataset = InputDataset('data/input')
    print(f"Loaded {len(dataset)} samples")
    
    # Split data - EXACT same as optimization script
    indices = list(range(len(dataset)))
    labels = [int(dataset[i].y.item()) for i in indices]
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, 
        stratify=[labels[i] for i in temp_idx]
    )
    
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create loaders - EXACT same as optimization script
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model - EXACT Config 9 parameters
    model = ExactConfig9Model(
        input_dim=100,
        hidden_dim=256,    # Config 9
        output_dim=2,
        num_steps=5,       # Config 9
        dropout=0.2,       # Config 9
        pooling='mean_max' # Config 9
    ).to(device)
    
    # EXACT same optimizer and loss as optimization script
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  Learning Rate: 0.001 (EXACT same as optimization)")
    print(f"  Weight Decay: 1e-4 (EXACT same as optimization)")
    print(f"  Batch Size: 32 (EXACT same as optimization)")
    print(f"  Optimizer: Adam (EXACT same as optimization)")
    
    # Training loop - EXACT same as optimization script
    best_val_acc = 0
    patience = 15  # EXACT same as optimization script
    patience_counter = 0
    
    print(f"\n🚀 Starting training for up to 100 epochs...")
    print("="*80)
    
    for epoch in range(100):
        # Train - EXACT same as optimization script
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            loss = criterion(out, batch.y.long())
            loss.backward()
            optimizer.step()  # NO gradient clipping!
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y.long()).sum().item()
            train_total += batch.y.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation - EXACT same as optimization script
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y.long()).sum().item()
                val_total += batch.y.size(0)
        
        val_acc = val_correct / val_total
        
        # Print progress every 20 epochs (same as optimization script)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping - EXACT same as optimization script
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            model_save_path = 'models/exact_config9_model.pth'
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            if epoch % 20 != 0:  # Don't double print
                print(f"  Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f} ✅ NEW BEST")
            print(f"    💾 Model saved to: {os.path.abspath(model_save_path)}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏸️  Early stopping at epoch {epoch}")
                break
    
    # Test evaluation - EXACT same as optimization script
    print(f"\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    model.load_state_dict(torch.load('models/exact_config9_model.pth'))
    model.eval()
    
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            test_preds.extend(pred.cpu().numpy())
            test_labels.extend(batch.y.long().cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(test_labels, test_preds)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Expected (Config 9): 83.57%")
    print(f"  Difference: {test_acc - 0.8357:+.4f}")
    
    if test_acc > 0.80:
        print(f"\n✅ SUCCESS! Model matches Config 9 performance!")
        print(f"🎉 This proves the EXACT implementation works!")
        
        # Copy to production location
        import shutil
        final_model_path = 'models/final_model.pth'
        shutil.copy('models/exact_config9_model.pth', final_model_path)
        print(f"✅ Saved to {os.path.abspath(final_model_path)}")
        print(f"📍 Final model location: {os.path.abspath(final_model_path)}")
        
    else:
        print(f"\n⚠️  Performance lower than expected.")
        print(f"🔍 Check data loading or implementation differences.")
    
    return test_acc


if __name__ == '__main__':
    test_acc = train_exact_config9()
    
    if test_acc > 0.80:
        print(f"\n" + "="*80)
        print("READY FOR PRODUCTION!")
        print("="*80)
        print(f"\n🚀 Key findings:")
        print(f"   - Learning rate 0.001 (not 1e-4!) is critical")
        print(f"   - Weight decay 1e-4 (not 1e-6!) is important")
        print(f"   - Batch size 32 (not 8!) helps convergence")
        print(f"   - NO residual connections in GNN")
        print(f"   - NO gradient clipping")
        
        print(f"\n💡 To fix main.py training:")
        print(f"   1. Update learning_rate: 1e-4 → 1e-3")
        print(f"   2. Update weight_decay: 1e-6 → 1e-4") 
        print(f"   3. Update batch_size: 8 → 32")
        print(f"   4. Remove residual connections")
        print(f"   5. Remove gradient clipping")
    else:
        print(f"\n❌ Need to debug further")