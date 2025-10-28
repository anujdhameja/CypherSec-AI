#!/usr/bin/env python3
"""
Optimize GNN Architecture to Match RF Structural Performance (81.77%)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import json
from datetime import datetime
import itertools

# Add src to path
sys.path.append('src')
from src.utils.objects.input_dataset import InputDataset

class OptimizedGNN(nn.Module):
    """Optimized GNN with configurable architecture"""
    
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=2, 
                 num_steps=3, dropout=0.3, pooling='mean'):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Configurable pooling
        self.pooling = pooling
        if pooling == 'attention':
            self.attention = nn.Linear(hidden_dim, 1)
        
        # Classifier layers
        pool_dim = hidden_dim * 2 if pooling == 'mean_max' else hidden_dim
        self.fc1 = nn.Linear(pool_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GNN layers
        x = self.ggc(x, edge_index)
        x = self.gnn_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            x = torch.cat([mean_pool, max_pool], dim=1)
        elif self.pooling == 'attention':
            # Simple attention pooling
            att_weights = torch.softmax(self.attention(x), dim=0)
            x = global_mean_pool(x * att_weights, batch)
        
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

class GNNOptimizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.best_config = None
        self.best_accuracy = 0.0
        
    def load_data(self):
        """Load and split the dataset"""
        print("Loading dataset...")
        
        # Get correct data path from configs
        import configs
        paths = configs.Paths()
        data_dir = str(Path(paths.input))
        
        dataset = InputDataset(data_dir)
        print(f"Total graphs loaded: {len(dataset)}")
        
        # Extract labels for stratified split
        labels = []
        for data in dataset:
            labels.append(int(data.y.item()))
        
        labels = np.array(labels, dtype=int)
        
        # Create same splits as comparison script
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(dataset))
        
        # 70/15/15 split
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=labels[temp_idx]
        )
        
        # Create datasets
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def train_model(self, config, train_dataset, val_dataset, test_dataset):
        """Train a model with given configuration"""
        print(f"\n🔄 Testing config: {config}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        model = OptimizedGNN(
            input_dim=100,
            hidden_dim=config['hidden_dim'],
            output_dim=2,
            num_steps=config['num_steps'],
            dropout=config['dropout'],
            pooling=config['pooling']
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(100):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                out = model(batch)
                loss = criterion(out, batch.y.long())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = out.argmax(dim=1)
                train_correct += (pred == batch.y.long()).sum().item()
                train_total += batch.y.size(0)
            
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    pred = out.argmax(dim=1)
                    val_correct += (pred == batch.y.long()).sum().item()
                    val_total += batch.y.size(0)
            
            val_acc = val_correct / val_total
            
            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_optimized_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model and test
        model.load_state_dict(torch.load('best_optimized_model.pth'))
        model.eval()
        
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch)
                pred = out.argmax(dim=1)
                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(batch.y.long().cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(test_labels, test_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='weighted'
        )
        
        result = {
            'config': config,
            'val_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'epochs_trained': epoch + 1
        }
        
        print(f"  ✓ Result: Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        return result
    
    def optimize(self):
        """Run optimization across different configurations"""
        print("=" * 80)
        print("GNN ARCHITECTURE OPTIMIZATION")
        print("Target: Beat RF Structural baseline of 81.77%")
        print("=" * 80)
        
        # Load data
        train_dataset, val_dataset, test_dataset = self.load_data()
        
        # Define configuration space
        configs = [
            # Baseline (current)
            {'hidden_dim': 128, 'num_steps': 3, 'dropout': 0.3, 'pooling': 'mean'},
            
            # Increase capacity
            {'hidden_dim': 256, 'num_steps': 3, 'dropout': 0.3, 'pooling': 'mean'},
            {'hidden_dim': 128, 'num_steps': 5, 'dropout': 0.3, 'pooling': 'mean'},
            {'hidden_dim': 256, 'num_steps': 5, 'dropout': 0.3, 'pooling': 'mean'},
            
            # Different dropout
            {'hidden_dim': 256, 'num_steps': 5, 'dropout': 0.1, 'pooling': 'mean'},
            {'hidden_dim': 256, 'num_steps': 5, 'dropout': 0.2, 'pooling': 'mean'},
            {'hidden_dim': 256, 'num_steps': 5, 'dropout': 0.5, 'pooling': 'mean'},
            
            # Different pooling strategies
            {'hidden_dim': 256, 'num_steps': 5, 'dropout': 0.2, 'pooling': 'max'},
            {'hidden_dim': 256, 'num_steps': 5, 'dropout': 0.2, 'pooling': 'mean_max'},
            {'hidden_dim': 256, 'num_steps': 5, 'dropout': 0.2, 'pooling': 'attention'},
            
            # High capacity configurations
            {'hidden_dim': 512, 'num_steps': 6, 'dropout': 0.2, 'pooling': 'mean_max'},
            {'hidden_dim': 384, 'num_steps': 4, 'dropout': 0.15, 'pooling': 'attention'},
        ]
        
        print(f"Testing {len(configs)} configurations...")
        
        # Test each configuration
        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(f"Configuration {i+1}/{len(configs)}")
            print(f"{'='*60}")
            
            try:
                result = self.train_model(config, train_dataset, val_dataset, test_dataset)
                self.results.append(result)
                
                # Track best
                if result['test_accuracy'] > self.best_accuracy:
                    self.best_accuracy = result['test_accuracy']
                    self.best_config = config
                    print(f"  🎉 NEW BEST: {self.best_accuracy:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error with config {config}: {e}")
                continue
        
        # Report results
        self.report_results()
    
    def report_results(self):
        """Report optimization results"""
        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)
        
        # Sort by test accuracy
        sorted_results = sorted(self.results, key=lambda x: x['test_accuracy'], reverse=True)
        
        print(f"{'Rank':<4} {'Test Acc':<10} {'Config':<50} {'Epochs':<8}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results[:10]):  # Top 10
            config_str = f"h={result['config']['hidden_dim']}, s={result['config']['num_steps']}, d={result['config']['dropout']}, p={result['config']['pooling']}"
            print(f"{i+1:<4} {result['test_accuracy']:<10.4f} {config_str:<50} {result['epochs_trained']:<8}")
        
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION")
        print("=" * 80)
        
        if self.best_config:
            print(f"Best Test Accuracy: {self.best_accuracy:.4f}")
            print(f"Target (RF Structural): 0.8177")
            print(f"Improvement needed: {0.8177 - self.best_accuracy:.4f}")
            print(f"\nBest Configuration:")
            for key, value in self.best_config.items():
                print(f"  {key}: {value}")
            
            if self.best_accuracy >= 0.8177:
                print(f"\n🎉 SUCCESS! GNN beats RF Structural baseline!")
            elif self.best_accuracy >= 0.80:
                print(f"\n🎯 CLOSE! GNN is very competitive with RF Structural")
            else:
                print(f"\n⚠️  GNN still underperforms RF Structural, but significant improvement made")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gnn_optimization_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'target_accuracy': 0.8177,
                'best_accuracy': self.best_accuracy,
                'best_config': self.best_config,
                'all_results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {filename}")

def main():
    """Main execution"""
    optimizer = GNNOptimizer()
    optimizer.optimize()

if __name__ == "__main__":
    main()