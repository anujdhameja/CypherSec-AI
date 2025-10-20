"""
Bayesian Hyperparameter Optimization for Devign Model
Uses scikit-optimize for intelligent hyperparameter search
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Import your existing modules
import configs
import src.data as data
import src.process as process
from src.process.balanced_training_config import BalancedDevignModel
from src.process.step import Step
from src.process.modeling import Train
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.callbacks import CheckpointSaver
except ImportError:
    print("ERROR: scikit-optimize not installed!")
    print("Install with: pip install scikit-optimize")
    exit(1)


class BayesianHyperparameterSearch:
    def __init__(self, n_calls=30, random_state=42):
        """
        Initialize Bayesian Hyperparameter Search
        
        Args:
            n_calls: Number of optimization iterations (trials)
            random_state: Random seed for reproducibility
        """
        self.n_calls = n_calls
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results tracking
        self.results = {
            'trials': [],
            'best_config': None,
            'best_score': -float('inf'),
            'search_space': None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load data once
        print("\n" + "="*80)
        print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        print(f"Using device: {self.device}")
        
        print("\n1. Loading data...")
        self._load_data()
        
    def _load_data(self):
        """Load and prepare datasets"""
        try:
            # Use the correct paths from configs
            PATHS = configs.Paths()
            
            # Load input datasets (using the current data loading approach)
            input_dataset = data.loads(PATHS.input)
            
            # Create a custom dataset class that works with DataFrames
            class DataFrameDataset:
                def __init__(self, dataframe):
                    self.data = dataframe['input'].tolist()
                    self.targets = dataframe['target'].tolist()
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    data_item = self.data[idx]
                    target = self.targets[idx]
                    
                    # Ensure target is in the data object
                    if not hasattr(data_item, 'y'):
                        data_item.y = torch.tensor([target], dtype=torch.long)
                    
                    return data_item
                
                def get_loader(self, batch_size=32, shuffle=True):
                    from torch_geometric.loader import DataLoader
                    from torch_geometric.data import Batch
                    return DataLoader(
                        self,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=lambda batch: Batch.from_data_list(batch)
                    )
            
            # Split the dataset manually (since train_val_test_split has issues)
            from sklearn.model_selection import train_test_split
            
            # First split: train + temp (80%) vs test (20%)
            train_temp, test_df = train_test_split(
                input_dataset, 
                test_size=0.2, 
                shuffle=True, 
                stratify=input_dataset['target'],
                random_state=42
            )
            
            # Second split: train (64%) vs val (16%)
            train_df, val_df = train_test_split(
                train_temp, 
                test_size=0.2, 
                shuffle=True, 
                stratify=train_temp['target'],
                random_state=42
            )
            
            # Create dataset objects
            self.train_dataset = DataFrameDataset(train_df)
            self.val_dataset = DataFrameDataset(val_df)
            self.test_dataset = DataFrameDataset(test_df)
            
            print(f"   Loaded {len(self.train_dataset)} training samples")
            print(f"   Loaded {len(self.val_dataset)} validation samples")
            print(f"   Loaded {len(self.test_dataset)} test samples")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def define_search_space(self):
        """
        Define the hyperparameter search space for Bayesian Optimization
        
        Returns:
            list: List of skopt dimension objects
        """
        search_space = [
            # Learning rate: log-uniform between 1e-5 and 1e-2
            Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
            
            # Weight decay: log-uniform between 1e-7 and 1e-4
            Real(1e-7, 1e-4, prior='log-uniform', name='weight_decay'),
            
            # Dropout: uniform between 0.1 and 0.5
            Real(0.1, 0.5, name='dropout'),
            
            # Number of GNN message passing steps: 2 to 6
            Integer(2, 6, name='num_steps'),
            
            # Hidden dimension: 128, 200, 256, 384
            Categorical([128, 200, 256, 384], name='hidden_dim'),
            
            # Batch size: 8, 16, 32 (minimum 8 to avoid BatchNorm issues)
            Categorical([8, 16, 32], name='batch_size'),
        ]
        
        # Store search space info
        self.results['search_space'] = {
            'learning_rate': 'log-uniform [1e-5, 1e-2]',
            'weight_decay': 'log-uniform [1e-7, 1e-4]',
            'dropout': 'uniform [0.1, 0.5]',
            'num_steps': 'integer [2, 6]',
            'hidden_dim': 'categorical [128, 200, 256, 384]',
            'batch_size': 'categorical [8, 16, 32]'
        }
        
        return search_space
    
    def train_and_evaluate(self, learning_rate, weight_decay, dropout, 
                          num_steps, hidden_dim, batch_size):
        """
        Train model with given hyperparameters and return validation score
        
        Returns:
            float: Negative validation accuracy (for minimization)
        """
        trial_num = len(self.results['trials']) + 1
        
        print("\n" + "="*80)
        print(f"TRIAL {trial_num}/{self.n_calls}")
        print("="*80)
        
        config = {
            'learning_rate': float(learning_rate),
            'weight_decay': float(weight_decay),
            'dropout': float(dropout),
            'num_steps': int(num_steps),
            'hidden_dim': int(hidden_dim),
            'batch_size': int(batch_size)
        }
        
        print(f"Config: {json.dumps(config, indent=2)}")
        
        start_time = time.time()
        
        try:
            # Use a BatchNorm-free version for hyperparameter optimization
            embed_config = configs.Embed()
            
            # Create a model without BatchNorm to avoid batch size issues
            class HyperoptDevignModel(nn.Module):
                def __init__(self, input_dim=100, output_dim=2, hidden_dim=200, 
                             num_steps=4, dropout=0.4):
                    super().__init__()
                    
                    # Input projection
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                    
                    # GNN
                    from torch_geometric.nn import GatedGraphConv, global_max_pool, global_mean_pool
                    self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
                    
                    # Classifier (NO BatchNorm to avoid batch size issues)
                    self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                    self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
                    
                    self.dropout = nn.Dropout(dropout)
                
                def forward(self, data):
                    x, edge_index, batch = data.x, data.edge_index, data.batch
                    
                    # Input
                    x = self.input_proj(x)
                    x = F.relu(x)
                    x = self.dropout(x)
                    
                    # GNN with residual
                    x_skip = x
                    x = self.ggc(x, edge_index)
                    x = F.relu(x + x_skip)  # Residual
                    x = self.dropout(x)
                    
                    # Dual pooling
                    from torch_geometric.nn import global_max_pool, global_mean_pool
                    x_mean = global_mean_pool(x, batch)
                    x_max = global_max_pool(x, batch)
                    x = torch.cat([x_mean, x_max], dim=1)
                    
                    # Classifier (no BatchNorm)
                    x = self.fc1(x)
                    x = F.relu(x)
                    x = self.dropout(x)
                    
                    x = self.fc2(x)
                    x = F.relu(x)
                    x = self.dropout(x)
                    
                    x = self.fc3(x)
                    
                    return x
            
            # Initialize model
            model = HyperoptDevignModel(
                input_dim=embed_config.nodes_dim,  # 100
                output_dim=2,  # Binary classification
                hidden_dim=config['hidden_dim'],
                num_steps=config['num_steps'],
                dropout=config['dropout']
            ).to(self.device)
            
            # Create optimizer and loss function (following current approach)
            
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
            
            # Create Step object (current training approach)
            step = Step(model=model, loss_function=loss_fn, optimizer=optimizer)
            
            # Create data loaders
            train_loader = self.train_dataset.get_loader(config['batch_size'], shuffle=True)
            val_loader = self.val_dataset.get_loader(config['batch_size'], shuffle=False)
            
            # Create loader steps
            train_loader_step = process.LoaderStep("Train", train_loader, self.device)
            val_loader_step = process.LoaderStep("Validation", val_loader, self.device)
            
            # Train with early stopping detection
            best_val_acc = 0.0
            stuck_count = 0
            max_stuck = 3
            epochs = 15  # Fixed epochs per trial
            
            for epoch in range(epochs):
                # Training phase
                try:
                    step.train()
                    train_stats = train_loader_step(step)
                    train_acc = train_stats.mean().acc * 100
                    train_loss = train_stats.loss()
                except RuntimeError as e:
                    if "Expected more than 1 value per channel" in str(e):
                        print(f"⚠️ BatchNorm error at epoch {epoch+1}, switching to eval mode")
                        step.eval()  # Use eval mode to avoid BatchNorm issues
                        train_stats = train_loader_step(step)
                        train_acc = train_stats.mean().acc * 100
                        train_loss = train_stats.loss()
                    else:
                        raise e
                
                # Validation phase
                with torch.no_grad():
                    step.eval()
                    val_stats = val_loader_step(step)
                    val_acc = val_stats.mean().acc * 100
                    val_loss = val_stats.loss()
                
                gap = train_acc - val_acc
                
                # Check if stuck (predicting one class)
                is_stuck = abs(val_acc - 55.56) < 1.0 or abs(val_acc - 44.44) < 1.0
                stuck_indicator = "⚠️ STUCK" if is_stuck else "✓"
                
                if is_stuck:
                    stuck_count += 1
                else:
                    stuck_count = 0
                
                print(f"Epoch {epoch+1:2d}: Train {train_acc:.2f}%, "
                      f"Val {val_acc:.2f}%, Gap {gap:+.2f}%, "
                      f"Loss {train_loss:.4f} {stuck_indicator}")
                
                # Early stopping if stuck
                if stuck_count >= max_stuck:
                    print(f"Stopping early - model stuck for {max_stuck} epochs")
                    break
                
                # Track best validation accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
            
            elapsed = time.time() - start_time
            
            print(f"\n📊 Trial Results:")
            print(f"   Best Val Acc: {best_val_acc:.2f}%")
            print(f"   Time: {elapsed:.1f}s")
            
            # Record trial
            trial_result = {
                'trial_num': trial_num,
                'config': config,
                'best_val_acc': best_val_acc,
                'final_train_acc': train_acc,
                'final_val_acc': val_acc,
                'time_seconds': elapsed,
                'converged': not is_stuck
            }
            
            self.results['trials'].append(trial_result)
            
            # Update best result
            if best_val_acc > self.results['best_score']:
                self.results['best_score'] = best_val_acc
                self.results['best_config'] = config
                print(f"   🎯 NEW BEST SCORE: {best_val_acc:.2f}%")
            
            # Save intermediate results
            self._save_results()
            
            # Return negative accuracy for minimization
            return -best_val_acc
            
        except Exception as e:
            print(f"❌ Error in trial: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failed trial
            trial_result = {
                'trial_num': trial_num,
                'config': config,
                'best_val_acc': 0.0,
                'error': str(e),
                'time_seconds': time.time() - start_time
            }
            self.results['trials'].append(trial_result)
            
            # Return worst possible score
            return 0.0
    
    def optimize(self):
        """
        Run Bayesian optimization to find best hyperparameters
        """
        print(f"\n2. Starting Bayesian Optimization ({self.n_calls} trials)...")
        print(f"   This will take approximately {self.n_calls * 2:.0f}-{self.n_calls * 3:.0f} minutes")
        
        # Define search space
        search_space = self.define_search_space()
        
        # Skip checkpoint callback to avoid pickling issues
        # checkpoint_dir = Path('hyperparameter_checkpoints')
        # checkpoint_dir.mkdir(exist_ok=True)
        # checkpoint_callback = CheckpointSaver(str(checkpoint_dir / "checkpoint.pkl"), compress=9)
        
        # Create objective function with named parameters
        @use_named_args(search_space)
        def objective(**params):
            return self.train_and_evaluate(**params)
        
        # Run Bayesian optimization
        try:
            result = gp_minimize(
                func=objective,
                dimensions=search_space,
                n_calls=self.n_calls,
                n_initial_points=min(5, self.n_calls),  # Adjust for small n_calls
                random_state=self.random_state,
                # callback=[checkpoint_callback],  # Removed to avoid pickling issues
                verbose=False
            )
            
            # Store optimization results
            self.results['optimization_result'] = {
                'best_score': -result.fun,  # Convert back to positive
                'best_params': dict(zip([dim.name for dim in search_space], result.x)),
                'n_calls': len(result.func_vals),
                'optimization_time': sum(t['time_seconds'] for t in self.results['trials'])
            }
            
            return result
            
        except KeyboardInterrupt:
            print("\n⚠️ Optimization interrupted by user")
            return None
    
    def _save_results(self):
        """Save results to JSON file"""
        output_file = Path('hyperparameter_results_bayesian.json')
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print optimization summary"""
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)
        
        if not self.results['trials']:
            print("No trials completed!")
            return
        
        # Sort trials by validation accuracy
        sorted_trials = sorted(
            self.results['trials'],
            key=lambda x: x.get('best_val_acc', 0),
            reverse=True
        )
        
        print(f"\n📊 Top 5 Configurations:\n")
        
        for i, trial in enumerate(sorted_trials[:5], 1):
            config = trial['config']
            acc = trial.get('best_val_acc', 0)
            
            print(f"{i}. Trial {trial['trial_num']}")
            print(f"   Best Val Acc: {acc:.2f}%")
            print(f"   Config:")
            print(f"      lr={config['learning_rate']:.2e}, wd={config['weight_decay']:.2e}")
            print(f"      dropout={config['dropout']:.3f}, steps={config['num_steps']}")
            print(f"      hidden_dim={config['hidden_dim']}, batch_size={config['batch_size']}")
            print(f"   Time: {trial['time_seconds']:.1f}s")
            print()
        
        # Statistics
        accuracies = [t.get('best_val_acc', 0) for t in self.results['trials'] if 'best_val_acc' in t]
        if accuracies:
            print(f"📈 Statistics:")
            print(f"   Best Val Acc: {max(accuracies):.2f}%")
            print(f"   Mean Val Acc: {np.mean(accuracies):.2f}%")
            print(f"   Std Val Acc: {np.std(accuracies):.2f}%")
            print(f"   Total Time: {sum(t['time_seconds'] for t in self.results['trials']):.1f}s")
        
        print(f"\n✓ Results saved to: hyperparameter_results_bayesian.json")


def main():
    """Main entry point"""
    # Configuration
    N_TRIALS = 30  # Number of Bayesian optimization iterations
    RANDOM_SEED = 42
    
    # Initialize search
    search = BayesianHyperparameterSearch(
        n_calls=N_TRIALS,
        random_state=RANDOM_SEED
    )
    
    # Run optimization
    result = search.optimize()
    
    # Print summary
    search.print_summary()
    
    print("\n" + "="*80)
    print("Bayesian optimization complete!")
    print("="*80)


if __name__ == "__main__":
    main()