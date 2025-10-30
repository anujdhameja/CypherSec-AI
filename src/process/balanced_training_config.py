"""
Balanced Training Configuration
Not too much overfitting, not too much underfitting - JUST RIGHT!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_max_pool, global_mean_pool


class BalancedDevignModel(nn.Module):
    """
    The Goldilocks Model - Balanced regularization
    
    Previous attempts:
    1. dropout=0.3, weight_decay=1.3e-6 → OVERFITTING (Train 89%, Val 49%)
    2. dropout=0.6, weight_decay=1e-4   → UNDERFITTING (Train 49%, Val 49%)
    3. THIS: dropout=0.4, weight_decay=1e-5 → BALANCED (hopefully!)
    """
    
    def __init__(self, input_dim=100, output_dim=2, hidden_dim=256, 
                 num_steps=5, dropout=0.2):
        super().__init__()
        
        print(f"\n=== BALANCED Devign Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        print(f"✓ Balanced regularization")
        print(f"   - GNN steps: {num_steps}")
        print(f"   - Dropout: {dropout}")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # GNN
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Dual pooling
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
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
        
        # GNN (simple - no residual like Config 9)
        x = self.ggc(x, edge_index)
        x = self.gnn_bn(x)
        x = F.relu(x)  # No residual connection
        x = self.dropout(x)
        
        # Dual pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
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
    
    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"💾 Model saved to: {os.path.abspath(path)}")
        print(f"📍 Model location: {os.path.abspath(path)}")
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"✓ Model loaded from {path}")


def goldilocks_config():
    """
    The Goldilocks configuration - not too much, not too little
    """
    config = {
        # Model parameters
        'input_dim': 205,
        'hidden_dim': 200,
        'output_dim': 2,
        'num_steps': 4,      # Not too deep (8), not too shallow (3)
        'dropout': 0.4,      # Not too low (0.3), not too high (0.6)
        
        # Training parameters
        'learning_rate': 2e-4,    # Slightly higher than 1e-4
        'weight_decay': 1e-5,     # Between 1.3e-6 and 1e-4
        'batch_size': 8,
        'epochs': 100,
        'patience': 15,           # More patience
        
        # Stability
        'gradient_clip': 1.0,
        'use_scheduler': True,
        'scheduler_patience': 7,
        'scheduler_factor': 0.5,
    }
    
    print("\n" + "="*80)
    print("GOLDILOCKS CONFIGURATION (Just Right!)")
    print("="*80)
    
    print("\n📊 Evolution of Configurations:")
    print("\n❌ Attempt 1 (Too Little Regularization):")
    print("   dropout=0.3, weight_decay=1.3e-6")
    print("   → Train 89%, Val 49% (OVERFITTING)")
    
    print("\n❌ Attempt 2 (Too Much Regularization):")
    print("   dropout=0.6, weight_decay=1e-4")
    print("   → Train 49%, Val 49% (UNDERFITTING - can't learn!)")
    
    print("\n✓ Attempt 3 (Balanced):")
    print(f"   dropout={config['dropout']}, weight_decay={config['weight_decay']:.0e}")
    print("   → Expected: Train 58-65%, Val 52-58% (BALANCED)")
    
    print("\n🎯 Key Changes from Previous Attempts:")
    print(f"   - Dropout: 0.6 → {config['dropout']} (relaxed)")
    print(f"   - Weight decay: 1e-4 → {config['weight_decay']:.0e} (reduced 10x)")
    print(f"   - Learning rate: 1e-4 → {config['learning_rate']:.0e} (increased)")
    print(f"   - GNN steps: 3 → {config['num_steps']} (slightly deeper)")
    
    print("\n📈 Expected Training Curve:")
    print("   Epoch 1-5:   Train 50-60%, Val 48-52%  (Learning starts)")
    print("   Epoch 10-20: Train 58-65%, Val 52-56%  (Improving)")
    print("   Epoch 30-50: Train 60-68%, Val 54-58%  (Converged)")
    print("   Gap:         5-10% is GOOD (not 40% like before!)")
    
    print("\n⚠️ Red Flags to Watch:")
    print("   ❌ Val acc stuck at 49.73% → Model predicting same class")
    print("   ❌ Train - Val gap > 20% → Too much overfitting")
    print("   ❌ Train acc < 55% after 20 epochs → Too much regularization")
    print("   ❌ Gradients > 50 → Still unstable")
    
    return config


def diagnose_stuck_validation():
    """
    Check why validation accuracy is stuck
    """
    print("\n" + "="*80)
    print("DIAGNOSING STUCK VALIDATION ACCURACY")
    print("="*80)
    
    print("\n🔍 Your Symptom:")
    print("   Val accuracy: 49.73% (EXACTLY, never changes)")
    
    print("\n💡 This means:")
    print("   Your model is predicting the SAME class for ALL validation samples")
    
    print("\n🧪 How to Verify:")
    print("""
# Add this to your validation loop:
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        output = model(batch)
        pred = output.argmax(dim=1)
        all_predictions.extend(pred.cpu().tolist())
        all_targets.extend(batch.y.squeeze().long().cpu().tolist())

print(f"Unique predictions: {set(all_predictions)}")
print(f"Prediction counts: {Counter(all_predictions)}")
print(f"Target distribution: {Counter(all_targets)}")

# If unique predictions is {0} or {1}, that's your problem!
""")
    
    print("\n🔧 Why This Happens:")
    print("   1. Model outputs are always very negative OR very positive")
    print("   2. Softmax/argmax always picks the same class")
    print("   3. Usually caused by:")
    print("      - Dead neurons (all outputs zero)")
    print("      - Exploding outputs (all very large)")
    print("      - Too much regularization (model gave up)")
    
    print("\n✅ How to Fix:")
    print("   1. Reduce dropout (0.6 → 0.4)")
    print("   2. Reduce weight decay (1e-4 → 1e-5)")
    print("   3. Increase learning rate slightly (1e-4 → 2e-4)")
    print("   4. Check model outputs before softmax:")
    print("""
# In validation, add:
print(f"Raw outputs: {output[:5]}")
print(f"After softmax: {F.softmax(output[:5], dim=1)}")
# If all outputs are similar, model is not discriminating
""")


def create_debug_validation_script():
    """
    Create script to debug why validation is stuck
    """
    script = '''
"""
Debug Validation Script
Check why validation accuracy is stuck at 49.73%
"""

import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from collections import Counter

# Load validation data (adjust path as needed)
df = pd.read_pickle('data/input/0_cpg_input.pkl')
val_size = int(0.1 * len(df))
val_data = df['input'].tolist()[-val_size:]

val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# Load your model
from src.process.model import DevignModel
model = DevignModel(input_dim=205, output_dim=2, max_edge_types=1, 
                   hidden_dim=200, num_steps=4, dropout=0.4)

# Try to load weights if available
try:
    model.load_state_dict(torch.load('data/model/devign.model'))
except:
    print("⚠️ No saved model, using random initialization")

model.eval()

all_predictions = []
all_targets = []
all_outputs = []

print("\\n" + "="*80)
print("VALIDATION PREDICTIONS DEBUG")
print("="*80)

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        output = model(batch)
        pred = output.argmax(dim=1)
        target = batch.y.squeeze().long()
        
        all_predictions.extend(pred.tolist())
        all_targets.extend(target.tolist())
        all_outputs.append(output)
        
        if i == 0:  # Show first batch details
            print(f"\\nFirst Batch Details:")
            print(f"Raw outputs:\\n{output}")
            print(f"Softmax outputs:\\n{torch.softmax(output, dim=1)}")
            print(f"Predictions: {pred.tolist()}")
            print(f"Targets: {target.tolist()}")

print(f"\\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

pred_counts = Counter(all_predictions)
target_counts = Counter(all_targets)

print(f"\\nPredictions distribution: {dict(pred_counts)}")
print(f"Target distribution: {dict(target_counts)}")

print(f"\\nUnique predictions: {set(all_predictions)}")

if len(set(all_predictions)) == 1:
    print("\\n❌ PROBLEM FOUND: Model predicts only ONE class!")
    print(f"   Always predicting: {list(set(all_predictions))[0]}")
    print(f"   This is why accuracy is stuck at {target_counts[list(set(all_predictions))[0]] / len(all_targets):.2%}")
else:
    print("\\n✓ Model is predicting both classes")

# Check output distribution
all_outputs_cat = torch.cat(all_outputs, dim=0)
print(f"\\nOutput statistics:")
print(f"   Mean: {all_outputs_cat.mean(dim=0)}")
print(f"   Std: {all_outputs_cat.std(dim=0)}")
print(f"   Min: {all_outputs_cat.min(dim=0)[0]}")
print(f"   Max: {all_outputs_cat.max(dim=0)[0]}")

print("\\n" + "="*80)
'''
    
    with open('debug_validation.py', 'w') as f:
        f.write(script)
    
    print("\n✓ Created debug_validation.py")
    print("Run: python debug_validation.py")


if __name__ == "__main__":
    # Show config
    config = goldilocks_config()
    
    # Diagnose issue
    diagnose_stuck_validation()
    
    # Create debug script
    create_debug_validation_script()
    
    print("\n" + "="*80)
    print("IMMEDIATE ACTION PLAN")
    print("="*80)
    print("\n1. Run: python debug_validation.py")
    print("   → This will show if model is predicting only one class")
    print("\n2. Update config with balanced settings:")
    print("   dropout = 0.4")
    print("   weight_decay = 1e-5")
    print("   learning_rate = 2e-4")
    print("\n3. Retrain and watch for:")
    print("   ✓ Val acc should CHANGE (not stuck at 49.73%)")
    print("   ✓ Train and Val should both improve")
    print("   ✓ Gap should be 5-15% (not 40%, not 0%)")