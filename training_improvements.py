"""
Training Improvements for Plateaued Performance
Apply these fixes if accuracy is stuck at 40-45%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_max_pool, global_mean_pool


# ============================================
# FIX 1: Add Learning Rate Scheduler
# ============================================

def create_scheduler(optimizer, config_type='step'):
    """
    Create learning rate scheduler to help escape plateaus
    
    config_type options:
    - 'step': Reduce LR every N epochs
    - 'plateau': Reduce LR when val loss plateaus
    - 'cosine': Cosine annealing
    """
    
    if config_type == 'step':
        # Reduce LR by 0.5 every 10 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=10, 
            gamma=0.5
        )
        print("✓ Using StepLR scheduler (LR × 0.5 every 10 epochs)")
    
    elif config_type == 'plateau':
        # Reduce LR when val loss stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        print("✓ Using ReduceLROnPlateau (LR × 0.5 after 5 epochs no improvement)")
    
    elif config_type == 'cosine':
        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
        print("✓ Using CosineAnnealingLR")
    
    else:
        scheduler = None
        print("✓ No scheduler")
    
    return scheduler


# Usage in training loop:
"""
scheduler = create_scheduler(optimizer, 'plateau')

for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)
    
    # Update scheduler
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)  # Plateau needs val_loss
    else:
        scheduler.step()  # Others just step
    
    # Print current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current LR: {current_lr:.2e}")
"""


# ============================================
# FIX 2: Improved Model with Residual Connections
# ============================================

class ImprovedDevignModel(nn.Module):
    """
    Enhanced model with:
    - Residual connections (helps gradient flow)
    - Batch normalization (stabilizes training)
    - Dropout (prevents overfitting)
    - Better pooling (mean + max)
    """
    
    def __init__(self, input_dim=205, hidden_dim=200, output_dim=2, 
                 num_steps=8, dropout=0.3):
        super().__init__()
        
        print(f"\n=== Improved Devign Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # GNN layers
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps)
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Pooling
        # Use both mean and max pooling for richer representation
        
        # Classifier with residual
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for mean+max
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✓ Architecture: {input_dim} → {hidden_dim} → {hidden_dim//2} → {output_dim}")
        print(f"✓ Using BatchNorm + Dropout({dropout})")
        print(f"✓ Dual pooling (mean + max)\n")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection with residual
        x_proj = F.relu(self.input_bn(self.input_proj(x)))
        
        # GNN
        x_gnn = self.ggc(x_proj, edge_index)
        x_gnn = self.gnn_bn(x_gnn)
        
        # Residual connection (if dims match)
        if x_proj.shape == x_gnn.shape:
            x = F.relu(x_proj + x_gnn)  # Residual
        else:
            x = F.relu(x_gnn)
        
        # Dual pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)  # [batch, hidden_dim * 2]
        
        # Classifier
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


# ============================================
# FIX 3: Gradient Clipping
# ============================================

def train_with_gradient_clipping(model, train_loader, optimizer, criterion, 
                                  device, max_grad_norm=1.0):
    """
    Training loop with gradient clipping to prevent exploding gradients
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        output = model(batch)
        target = batch.y.squeeze().long()
        
        # Loss
        loss = criterion(output, target)
        
        # Backward
        loss.backward()
        
        # Gradient clipping (prevents explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Stats
        total_loss += loss.item() * batch.num_graphs
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += batch.num_graphs
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


# ============================================
# FIX 4: Better Weight Initialization
# ============================================

def initialize_weights(model):
    """
    Proper weight initialization can help training
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    print("✓ Initialized weights with Xavier uniform")


# ============================================
# FIX 5: Data Augmentation for Graphs
# ============================================

def augment_graph(data, drop_edge_prob=0.1):
    """
    Simple graph augmentation: randomly drop edges during training
    This acts as regularization
    """
    if drop_edge_prob == 0:
        return data
    
    edge_index = data.edge_index
    num_edges = edge_index.shape[1]
    
    # Keep edges with probability (1 - drop_edge_prob)
    mask = torch.rand(num_edges) > drop_edge_prob
    
    data.edge_index = edge_index[:, mask]
    
    return data


# ============================================
# FIX 6: Focal Loss for Imbalanced Data
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss: focuses on hard examples
    Useful if data is imbalanced or model plateaus
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ============================================
# COMPLETE TRAINING SCRIPT WITH ALL FIXES
# ============================================

def improved_training_loop():
    """
    Complete training loop with all improvements
    """
    print("="*80)
    print("IMPROVED TRAINING SETUP")
    print("="*80)
    
    # 1. Model with improvements
    model = ImprovedDevignModel(
        input_dim=205,
        hidden_dim=200,
        output_dim=2,
        num_steps=8,
        dropout=0.3
    )
    initialize_weights(model)
    
    # 2. Loss function (try focal loss if regular CE doesn't work)
    use_focal = False
    if use_focal:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("✓ Using Focal Loss (for hard examples)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("✓ Using CrossEntropy Loss")
    
    # 3. Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1.3e-6
    )
    print(f"✓ Adam optimizer (lr=1e-4)")
    
    # 4. Scheduler
    scheduler = create_scheduler(optimizer, 'plateau')
    
    # 5. Training loop
    """
    for epoch in range(num_epochs):
        # Training with gradient clipping
        train_loss, train_acc = train_with_gradient_clipping(
            model, train_loader, optimizer, criterion, device, max_grad_norm=1.0
        )
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        print(f"Epoch {epoch}: Train Acc={train_acc:.2%}, Val Acc={val_acc:.2%}")
    """


# ============================================
# DIAGNOSTIC: Check What's Wrong
# ============================================

def diagnose_plateau(train_accs, val_accs, train_losses, val_losses):
    """
    Diagnose why model plateaued
    """
    print("\n" + "="*80)
    print("PLATEAU DIAGNOSIS")
    print("="*80)
    
    # Check if overfitting
    final_train_acc = train_accs[-1]
    final_val_acc = val_accs[-1]
    
    if final_train_acc - final_val_acc > 0.15:
        print("\n⚠️ OVERFITTING DETECTED")
        print(f"   Train: {final_train_acc:.2%}, Val: {final_val_acc:.2%}")
        print("   Solutions:")
        print("   - Increase dropout (try 0.5)")
        print("   - Add more regularization")
        print("   - Use data augmentation")
        print("   - Reduce model complexity")
    
    # Check if underfitting
    elif final_train_acc < 0.6:
        print("\n⚠️ UNDERFITTING DETECTED")
        print(f"   Train: {final_train_acc:.2%}")
        print("   Solutions:")
        print("   - Increase learning rate (try 5e-4)")
        print("   - Increase model capacity")
        print("   - Train for more epochs")
        print("   - Check data quality")
    
    # Check if plateau
    else:
        recent_val = val_accs[-5:]
        if max(recent_val) - min(recent_val) < 0.02:
            print("\n⚠️ PLATEAUED")
            print("   Solutions:")
            print("   - Use learning rate scheduler")
            print("   - Try different optimizer (SGD with momentum)")
            print("   - Add batch normalization")
            print("   - Use residual connections")


if __name__ == "__main__":
    print(__doc__)
    improved_training_loop()