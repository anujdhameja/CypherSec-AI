#!/usr/bin/env python3
"""
Fair Model Comparison Script
Compares GNN vs Random Forest baselines on identical train/val/test splits
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pickle
import json
from datetime import datetime

# Add src to path
sys.path.append('src')
from src.utils.objects.input_dataset import InputDataset
from src.process.balanced_training_config import BalancedDevignModel
import configs
from pathlib import Path

class ModelComparison:
    def __init__(self, random_state=42):
        # Get correct data path from configs
        paths = configs.Paths()
        self.data_dir = str(Path(paths.input))
        self.random_state = random_state
        self.results = {}
        
    def load_full_dataset(self):
        """Load the complete dataset with all 1893 graphs"""
        print("Loading full dataset...")
        
        # Load dataset
        dataset = InputDataset(self.data_dir)
        print(f"Total graphs loaded: {len(dataset)}")
        
        # Extract labels
        labels = []
        for data in dataset:
            labels.append(int(data.y.item()))
        
        labels = np.array(labels, dtype=int)
        print(f"Label distribution: {np.bincount(labels)}")
        
        return dataset, labels 
   
    def create_splits(self, dataset, labels):
        """Create 70/15/15 train/val/test splits"""
        print("Creating train/val/test splits (70/15/15)...")
        
        indices = np.arange(len(dataset))
        
        # First split: 70% train, 30% temp
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=self.random_state, 
            stratify=labels
        )
        
        # Second split: 15% val, 15% test from the 30% temp
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=self.random_state,
            stratify=labels[temp_idx]
        )
        
        print(f"Train size: {len(train_idx)} ({len(train_idx)/len(dataset)*100:.1f}%)")
        print(f"Val size: {len(val_idx)} ({len(val_idx)/len(dataset)*100:.1f}%)")
        print(f"Test size: {len(test_idx)} ({len(test_idx)/len(dataset)*100:.1f}%)")
        
        # Verify label distributions
        for split_name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
            split_labels = labels[idx]
            dist = np.bincount(split_labels)
            percentages = dist/len(split_labels)*100
            print(f"{split_name} label distribution: {dist} ({percentages[0]:.1f}%, {percentages[1]:.1f}%)")
        
        return train_idx, val_idx, test_idx
    
    def extract_node_features(self, dataset, indices):
        """Extract node features for Random Forest (sample nodes from each graph)"""
        print("Extracting node features for Random Forest...")
        
        features_list = []
        labels_list = []
        
        for idx in indices:
            data = dataset[idx]
            graph_label = data.y.item()
            
            # Sample up to 10 nodes per graph to keep it manageable
            num_nodes = data.x.shape[0]
            if num_nodes > 10:
                node_indices = np.random.choice(num_nodes, 10, replace=False)
            else:
                node_indices = np.arange(num_nodes)
            
            for node_idx in node_indices:
                node_features = data.x[node_idx].numpy()
                features_list.append(node_features)
                labels_list.append(graph_label)
        
        return np.array(features_list), np.array(labels_list)    

    def extract_structural_features(self, dataset, indices):
        """Extract graph-level structural features"""
        print("Extracting structural features for Random Forest...")
        
        features_list = []
        labels_list = []
        
        for idx in indices:
            data = dataset[idx]
            
            # Basic structural features
            num_nodes = data.x.shape[0]
            num_edges = data.edge_index.shape[1]
            
            # Edge density
            max_edges = num_nodes * (num_nodes - 1)
            edge_density = num_edges / max_edges if max_edges > 0 else 0
            
            # Average degree
            avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
            
            # Node feature statistics
            node_feat_mean = data.x.mean(dim=0).numpy()
            node_feat_std = data.x.std(dim=0).numpy()
            node_feat_max = data.x.max(dim=0)[0].numpy()
            node_feat_min = data.x.min(dim=0)[0].numpy()
            
            # Combine all features
            structural_features = np.concatenate([
                [num_nodes, num_edges, edge_density, avg_degree],
                node_feat_mean,
                node_feat_std,
                node_feat_max,
                node_feat_min
            ])
            
            features_list.append(structural_features)
            labels_list.append(data.y.item())
        
        return np.array(features_list), np.array(labels_list)   
 
    def train_gnn(self, dataset, train_idx, val_idx, test_idx):
        """Train the GNN model"""
        print("\n=== Training GNN Model ===")
        
        # Create data loaders
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Get feature dimension from first sample
        sample_data = dataset[0]
        input_dim = sample_data.x.shape[1]
        
        model = BalancedDevignModel(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=2,
            num_steps=3,
            dropout=0.3
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Train
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
                optimizer.step()
                
                train_loss += loss.item()
                pred = out.argmax(dim=1)
                train_correct += (pred == batch.y).sum().item()
                train_total += batch.y.size(0)
            
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    pred = out.argmax(dim=1)
                    val_correct += (pred == batch.y).sum().item()
                    val_total += batch.y.size(0)
            
            val_acc = val_correct / val_total
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_gnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break 
       
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load('best_gnn_model.pth'))
        model.eval()
        
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        test_acc = accuracy_score(test_labels, test_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='weighted'
        )
        
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())
        
        val_acc_final = accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted'
        )
        
        return {
            'val_accuracy': val_acc_final,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'test_accuracy': test_acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        }    

    def train_rf_node_features(self, dataset, train_idx, val_idx, test_idx):
        """Train Random Forest on node features"""
        print("\n=== Training Random Forest (Node Features) ===")
        
        # Extract features
        X_train, y_train = self.extract_node_features(dataset, train_idx)
        X_val, y_val = self.extract_node_features(dataset, val_idx)
        X_test, y_test = self.extract_node_features(dataset, test_idx)
        
        print(f"Node features shape: {X_train.shape}")
        print(f"Training samples: {len(X_train)}")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Predictions
        val_preds = rf.predict(X_val)
        test_preds = rf.predict(X_test)
        
        # Metrics
        val_acc = accuracy_score(y_val, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='weighted'
        )
        
        test_acc = accuracy_score(y_test, test_preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, test_preds, average='weighted'
        )
        
        return {
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
    
    def train_rf_structural_features(self, dataset, train_idx, val_idx, test_idx):
        """Train Random Forest on structural features"""
        print("\n=== Training Random Forest (Structural Features) ===")
        
        # Extract features
        X_train, y_train = self.extract_structural_features(dataset, train_idx)
        X_val, y_val = self.extract_structural_features(dataset, val_idx)
        X_test, y_test = self.extract_structural_features(dataset, test_idx)
        
        print(f"Structural features shape: {X_train.shape}")
        print(f"Training samples: {len(X_train)}")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Predictions
        val_preds = rf.predict(X_val)
        test_preds = rf.predict(X_test)
        
        # Metrics
        val_acc = accuracy_score(y_val, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average='weighted'
        )
        
        test_acc = accuracy_score(y_test, test_preds)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, test_preds, average='weighted'
        )
        
        return {
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
    
    def run_comparison(self):
        """Run the complete model comparison"""
        print("=" * 60)
        print("FAIR MODEL COMPARISON")
        print("=" * 60)
        
        # Load dataset and create splits
        dataset, labels = self.load_full_dataset()
        train_idx, val_idx, test_idx = self.create_splits(dataset, labels)
        
        # Train all models
        gnn_results = self.train_gnn(dataset, train_idx, val_idx, test_idx)
        rf_node_results = self.train_rf_node_features(dataset, train_idx, val_idx, test_idx)
        rf_struct_results = self.train_rf_structural_features(dataset, train_idx, val_idx, test_idx)
        
        # Store results
        self.results = {
            'GNN (BalancedDevignModel)': gnn_results,
            'Random Forest (Node Features)': rf_node_results,
            'Random Forest (Structural Features)': rf_struct_results
        }
        
        # Print comparison
        self.print_results()
        
        # Save results
        self.save_results()  
  
    def print_results(self):
        """Print formatted comparison results"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        print(f"{'Model':<35} {'Val Acc':<10} {'Test Acc':<10} {'Test P':<10} {'Test R':<10} {'Test F1':<10}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<35} "
                  f"{results['val_accuracy']:<10.4f} "
                  f"{results['test_accuracy']:<10.4f} "
                  f"{results['test_precision']:<10.4f} "
                  f"{results['test_recall']:<10.4f} "
                  f"{results['test_f1']:<10.4f}")
        
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        
        # Find best performing model
        best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"Best performing model: {best_model[0]} (Test Acc: {best_model[1]['test_accuracy']:.4f})")
        
        # Compare GNN vs baselines
        gnn_acc = self.results['GNN (BalancedDevignModel)']['test_accuracy']
        rf_node_acc = self.results['Random Forest (Node Features)']['test_accuracy']
        rf_struct_acc = self.results['Random Forest (Structural Features)']['test_accuracy']
        
        print(f"\nGNN vs Node Features RF: {gnn_acc:.4f} vs {rf_node_acc:.4f} "
              f"({'GNN wins' if gnn_acc > rf_node_acc else 'RF wins'} by {abs(gnn_acc - rf_node_acc):.4f})")
        
        print(f"GNN vs Structural RF: {gnn_acc:.4f} vs {rf_struct_acc:.4f} "
              f"({'GNN wins' if gnn_acc > rf_struct_acc else 'RF wins'} by {abs(gnn_acc - rf_struct_acc):.4f})")
        
        if gnn_acc < max(rf_node_acc, rf_struct_acc):
            print("\n⚠️  WARNING: GNN is underperforming compared to simple baselines!")
            print("This suggests the GNN architecture or training needs optimization.")
        else:
            print("\n✅ GNN is outperforming baseline models as expected.")
    
    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_results_{timestamp}.json"
        
        # Add metadata
        results_with_metadata = {
            'timestamp': timestamp,
            'random_state': self.random_state,
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to: {filename}")

def main():
    """Main execution function"""
    print("Starting fair model comparison...")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run comparison
    comparison = ModelComparison()
    comparison.run_comparison()

if __name__ == "__main__":
    main()