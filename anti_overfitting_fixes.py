"""
Anti-Overfitting Configuration
Fixes severe overfitting (89% train vs 49% val)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_max_pool, global_mean_pool


class RegularizedDevignModel(nn.Module):
    """
    Heavily regularized model to prevent overfitting
    
    Key changes:
    1. MUCH higher dropout (0.3 → 0.6)
    2. L2 regularization
    3. Dual pooling (mean + max)
    4. Simpler architecture
    """
    
    def __init__(self, input_dim=205, output_dim=2, hidden_dim=200, 
                 num_steps=3, dropout=0.6):  # INCREASED dropout, REDUCED steps
        super().__init__()
        
        print(f"\n=== REGULARIZED Devign Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        print(f"⚠️ Heavy regularization applied!")
        print(f"   - GNN steps: {num_steps} (reduced)")
        print(f"   - Dropout: {dropout} (doubled!)")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # GNN with MINIMAL steps to prevent overfitting
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Dual pooling for better generalization
        # Using BOTH mean and max captures more diverse features
        
        # Simple classifier with HEAVY dropout
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for dual pooling
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # CRITICAL: Much higher dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"✓ Using dual pooling (mean + max)")
        print(f"✓ Dropout rate: {dropout}")
        print("="*50 + "\n")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout even at input!
        
        # GNN
        x = self.ggc(x, edge_index)
        x = self.gnn_bn(x)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout after GNN
        
        # Dual pooling (more robust than single pooling)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classifier with heavy dropout
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)  # Dropout before final layer
        
        x = self.fc2(x)
        
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"✓ Model loaded from {path}")


def anti_overfitting_config():
    """
    Configuration specifically designed to prevent overfitting
    """
    config = {
        # Model parameters - SIMPLIFIED
        'input_dim': 205,
        'hidden_dim': 200,
        'output_dim': 2,
        'num_steps': 3,  # REDUCED from 4 to 3
        'dropout': 0.6,  # DOUBLED from 0.3
        
        # Training parameters
        'learning_rate': 1e-4,  # Back to original (slower = less overfitting)
        'weight_decay': 1e-4,   # INCREASED from 1.3e-6 (100x more L2 reg!)
        'batch_size': 8,
        'epochs': 50,
        'patience': 5,  # Stop earlier if no improvement
        
        # Stability parameters
        'gradient_clip': 0.5,  # REDUCED from 1.0 (more aggressive clipping)
        'use_scheduler': True,
        'scheduler_patience': 3,
        'scheduler_factor': 0.5,
        
        # NEW: Early stopping on validation accuracy
        'early_stop_metric': 'val_acc',  # Stop when val acc stops improving
        'early_stop_patience': 5,
    }
    
    print("\n" + "="*80)
    print("ANTI-OVERFITTING CONFIGURATION")
    print("="*80)
    
    print("\n🛡️ Regularization Techniques Applied:")
    print(f"   1. Dropout: 0.3 → {config['dropout']} (doubled!)")
    print(f"   2. Weight decay: 1.3e-6 → {config['weight_decay']:.0e} (100x stronger!)")
    print(f"   3. GNN steps: 4 → {config['num_steps']} (reduced complexity)")
    print(f"   4. Gradient clip: 1.0 → {config['gradient_clip']}")
    print(f"   5. Early stopping: patience={config['early_stop_patience']}")
    print(f"   6. Dual pooling: mean + max (more robust)")
    
    print("\n🎯 Expected Performance (with proper regularization):")
    print("   - Epoch 1-5:   Train ~50-60%, Val ~45-50%")
    print("   - Epoch 10-20: Train ~60-65%, Val ~52-56%")
    print("   - Epoch 30+:   Train ~62-68%, Val ~54-58%")
    print("\n   ✓ Train and Val should be CLOSE (within 5-10%)")
    print("   ✓ Val loss should DECREASE, not explode!")
    
    print("\n⚠️ Warning Signs to Watch:")
    print("   ❌ Train acc > 70% while Val acc < 55% = OVERFITTING")
    print("   ❌ Val loss increasing = OVERFITTING")
    print("   ❌ Gradient norms > 50 = Still unstable")
    
    return config


def train_with_validation_monitoring(model, train_loader, val_loader, optimizer, 
                                    criterion, device, num_epochs, patience=5):
    """
    Training loop with proper validation monitoring and early stopping
    """
    best_val_acc = 0
    patience_counter = 0
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch)
            target = batch.y.squeeze().long()
            
            loss = criterion(output, target)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += batch.num_graphs
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                target = batch.y.squeeze().long()
                
                loss = criterion(output, target)
                if not torch.isnan(loss):
                    val_loss += loss.item() * batch.num_graphs
                    pred = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total += batch.num_graphs
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        # Print with overfitting warning
        gap = train_acc - val_acc
        warning = "⚠️ OVERFITTING!" if gap > 0.15 else "✓" if gap < 0.1 else ""
        
        print(f"Epoch {epoch+1:3d}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%} | "
              f"Gap: {gap:+.2%} {warning}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered (no improvement for {patience} epochs)")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print(f"\n✓ Loaded best model (val_acc={best_val_acc:.2%})")
    
    return model


def compare_configurations():
    """Compare old vs new configuration"""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    print("\n📊 OLD Configuration (Overfitting):")
    print("   Dropout: 0.3")
    print("   Weight Decay: 1.3e-6")
    print("   GNN Steps: 4")
    print("   Result: Train 89%, Val 49% ❌")
    
    print("\n📊 NEW Configuration (Regularized):")
    print("   Dropout: 0.6 (2x)")
    print("   Weight Decay: 1e-4 (100x)")
    print("   GNN Steps: 3")
    print("   Expected: Train 60-65%, Val 54-58% ✓")
    
    print("\n💡 Key Insight:")
    print("   High training accuracy is NOT always good!")
    print("   We want train and val to be CLOSE.")
    print("   Better: Train 62%, Val 58% than Train 89%, Val 49%")


if __name__ == "__main__":
    # Show config
    config = anti_overfitting_config()
    
    # Compare
    compare_configurations()
    
    print("\n🎯 IMMEDIATE ACTION:")
    print("1. Replace model with RegularizedDevignModel")
    print("2. Set dropout=0.6, weight_decay=1e-4")
    print("3. Reduce num_steps to 3")
    print("4. Monitor that train and val stay close!")
    print("\nExpect LOWER train acc (~60-65%) but HIGHER val acc (~54-58%)")