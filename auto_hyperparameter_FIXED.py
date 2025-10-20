"""
FIXED Comprehensive Hyperparameter Search
Works with your updated codebase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, global_mean_pool
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
from collections import Counter
import numpy as np


# ============================================
# Simple Model (for hyperparameter search)
# ============================================

class SimpleBalancedModel(nn.Module):
    """Simplified model for faster hyperparameter search"""
    
    def __init__(self, input_dim=100, hidden_dim=200, output_dim=2, 
                 num_steps=4, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
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
        
        # Dual pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_mean_pool(x, batch)  # Using mean twice to avoid NaN
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classifier
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x


# ============================================
# Training Functions
# ============================================

def train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        output = model(batch)
        target = batch.y.squeeze().long()
        
        loss = criterion(output, target)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += batch.num_graphs
    
    return total_loss / total, correct / total


def validate(model, val_loader, criterion, device):
    """Validate"""
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
    
    # Check diversity
    unique_preds = len(set(all_preds))
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'unique_predictions': unique_preds,
        'predictions': Counter(all_preds),
        'targets': Counter(all_targets)
    }


# ============================================
# Hyperparameter Configurations
# ============================================

def get_search_configs():
    """Define configurations to search"""
    
    configs = [
        # Baseline
        {
            'name': 'baseline',
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'dropout': 0.2,
            'num_steps': 4,
            'hidden_dim': 200,
        },
        
        # Higher learning rate
        {
            'name': 'higher_lr',
            'learning_rate': 3e-4,
            'weight_decay': 1e-6,
            'dropout': 0.2,
            'num_steps': 4,
            'hidden_dim': 200,
        },
        
        # More regularization
        {
            'name': 'more_reg',
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'dropout': 0.3,
            'num_steps': 3,
            'hidden_dim': 200,
        },
        
        # Simpler model
        {
            'name': 'simple',
            'learning_rate': 2e-4,
            'weight_decay': 1e-6,
            'dropout': 0.15,
            'num_steps': 3,
            'hidden_dim': 150,
        },
        
        # Larger model
        {
            'name': 'larger',
            'learning_rate': 1e-4,
            'weight_decay': 5e-6,
            'dropout': 0.25,
            'num_steps': 5,
            'hidden_dim': 256,
        },
    ]
    
    return configs


# ============================================
# Experiment Runner
# ============================================

def run_experiment(config, train_loader, val_loader, device, max_epochs=30):
    """Run single experiment"""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {config['name']}")
    print(f"{'='*80}")
    print(f"Config: {config}")
    
    # Create model
    model = SimpleBalancedModel(
        input_dim=100,
        hidden_dim=config['hidden_dim'],
        output_dim=2,
        num_steps=config['num_steps'],
        dropout=config['dropout']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Loss with class weights
    # Calculate from train_loader
    all_targets = []
    for batch in train_loader:
        all_targets.extend(batch.y.squeeze().long().tolist())
    
    target_counts = Counter(all_targets)
    total = sum(target_counts.values())
    class_weights = torch.tensor([
        total / (2 * target_counts[0]),
        total / (2 * target_counts[1])
    ], dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    patience = 10
    
    results = {
        'config': config,
        'train_accs': [],
        'val_accs': [],
        'train_losses': [],
        'val_losses': [],
    }
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_results = validate(model, val_loader, criterion, device)
        val_loss = val_results['loss']
        val_acc = val_results['accuracy']
        
        results['train_accs'].append(train_acc)
        results['val_accs'].append(val_acc)
        results['train_losses'].append(train_loss)
        results['val_losses'].append(val_loss)
        
        # Print
        gap = train_acc - val_acc
        status = "✓" if val_results['unique_predictions'] == 2 else "⚠️ STUCK"
        
        print(f"Epoch {epoch+1:2d}: "
              f"Train {train_acc:.2%}, Val {val_acc:.2%}, "
              f"Gap {gap:+.2%}, Loss {val_loss:.4f} {status}")
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Emergency stop if stuck
        if epoch > 5 and val_results['unique_predictions'] == 1:
            print(f"Stopping - model stuck predicting one class")
            break
    
    elapsed = time.time() - start_time
    
    # Final results
    results['best_val_acc'] = best_val_acc
    results['best_epoch'] = best_epoch
    results['final_train_acc'] = results['train_accs'][-1]
    results['final_val_acc'] = results['val_accs'][-1]
    results['time_seconds'] = elapsed
    
    print(f"\n📊 Results:")
    print(f"   Best Val Acc: {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"   Final Train/Val: {results['final_train_acc']:.2%} / {results['final_val_acc']:.2%}")
    print(f"   Time: {elapsed:.1f}s")
    
    return results


# ============================================
# Main
# ============================================

def main():
    print("\n" + "="*80)
    print("SIMPLIFIED HYPERPARAMETER SEARCH")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading data...")
    input_path = Path('data/input')
    input_files = list(input_path.glob('*_input.pkl'))
    
    if not input_files:
        print("❌ No input files found!")
        return
    
    # Load first few files for faster search
    all_data = []
    for f in input_files[:10]:  # Use subset for faster search
        df = pd.read_pickle(f)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    graphs = combined['input'].tolist()
    
    print(f"   Loaded {len(graphs)} samples")
    
    # Split
    train_size = int(0.8 * len(graphs))
    val_size = int(0.1 * len(graphs))
    
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:train_size+val_size]
    
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
    
    print(f"   Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    
    # Get configurations
    configs = get_search_configs()
    print(f"\n2. Testing {len(configs)} configurations...")
    
    # Run experiments
    all_results = []
    
    for config in configs:
        try:
            results = run_experiment(
                config, train_loader, val_loader, device, max_epochs=20
            )
            all_results.append(results)
            
            # Save after each experiment
            with open('hyperparameter_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in experiment: {e}")
            continue
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_results:
        sorted_results = sorted(
            all_results,
            key=lambda x: x['best_val_acc'],
            reverse=True
        )
        
        print("\n📊 Top 3 Configurations:")
        for i, res in enumerate(sorted_results[:3], 1):
            print(f"\n{i}. {res['config']['name']}")
            print(f"   Best Val Acc: {res['best_val_acc']:.2%}")
            print(f"   Config: lr={res['config']['learning_rate']:.0e}, "
                  f"wd={res['config']['weight_decay']:.0e}, "
                  f"dropout={res['config']['dropout']}, "
                  f"steps={res['config']['num_steps']}")
        
        print(f"\n✓ Results saved to: hyperparameter_results.json")
    else:
        print("\n⚠️ No experiments completed")


if __name__ == "__main__":
    main()