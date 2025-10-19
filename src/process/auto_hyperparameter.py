"""
Automated Hyperparameter Search
Systematically tests configurations and logs results
FREE and runs locally - no external services needed!
"""

import torch
import torch.nn as nn
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import product
import pandas as pd


class ExperimentLogger:
    """Logs all experiments to a file for analysis"""
    
    def __init__(self, log_file='experiments_log.json'):
        self.log_file = log_file
        self.experiments = []
        self.load_existing()
    
    def load_existing(self):
        """Load existing experiments if file exists"""
        if Path(self.log_file).exists():
            with open(self.log_file, 'r') as f:
                self.experiments = json.load(f)
            print(f"✓ Loaded {len(self.experiments)} existing experiments")
    
    def log_experiment(self, config, results):
        """Log a single experiment"""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        }
        self.experiments.append(experiment)
        
        # Save after each experiment
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        print(f"\n✓ Logged experiment #{len(self.experiments)}")
    
    def get_best_experiments(self, n=5, metric='best_val_acc'):
        """Get top N experiments by metric"""
        sorted_exp = sorted(
            self.experiments,
            key=lambda x: x['results'].get(metric, 0),
            reverse=True
        )
        return sorted_exp[:n]
    
    def print_summary(self):
        """Print summary of all experiments"""
        if not self.experiments:
            print("No experiments yet!")
            return
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        # Convert to DataFrame for easy analysis
        data = []
        for exp in self.experiments:
            row = {**exp['config'], **exp['results']}
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Show best configurations
        print("\n📊 Top 5 Configurations by Validation Accuracy:")
        top5 = df.nlargest(5, 'best_val_acc')[
            ['dropout', 'weight_decay', 'learning_rate', 'num_steps', 
             'best_val_acc', 'final_train_acc', 'converged_epoch']
        ]
        print(top5.to_string(index=False))
        
        # Show correlation analysis
        print("\n📈 Configuration Impact Analysis:")
        numeric_cols = ['dropout', 'weight_decay', 'learning_rate', 'num_steps']
        for col in numeric_cols:
            corr = df[col].corr(df['best_val_acc'])
            print(f"   {col:15s} → Val Acc correlation: {corr:+.3f}")


class EarlyStopChecker:
    """Detects when experiment should be stopped early"""
    
    def __init__(self, patience=5, min_epochs=10):
        self.patience = patience
        self.min_epochs = min_epochs
    
    def should_stop(self, epoch, train_acc, val_acc, val_loss_history):
        """
        Stop if:
        1. Val acc stuck (not changing)
        2. Severe overfitting (train >> val)
        3. Val loss exploding
        """
        # Must run minimum epochs
        if epoch < self.min_epochs:
            return False, None
        
        # Check 1: Val acc stuck (hasn't changed in last 5 epochs)
        recent_epochs = 5
        if epoch >= recent_epochs:
            recent_vals = val_loss_history[-recent_epochs:]
            if len(set(recent_vals)) == 1:  # All same
                return True, f"Val acc stuck at {val_acc:.2%} for {recent_epochs} epochs"
        
        # Check 2: Severe overfitting
        gap = train_acc - val_acc
        if gap > 0.30:  # 30% gap
            return True, f"Severe overfitting (gap={gap:.2%})"
        
        # Check 3: Val loss exploding
        if len(val_loss_history) >= 3:
            recent_increase = val_loss_history[-1] / val_loss_history[-3]
            if recent_increase > 2.0:  # Loss doubled
                return True, f"Val loss exploding ({val_loss_history[-3]:.2f} → {val_loss_history[-1]:.2f})"
        
        return False, None


def grid_search_configs():
    """
    Define grid search space
    
    Strategy: Test combinations systematically
    """
    search_space = {
        'dropout': [0.35, 0.4, 0.45, 0.5],
        'weight_decay': [1e-6, 5e-6, 1e-5, 5e-5],
        'learning_rate': [1e-4, 2e-4, 3e-4],
        'num_steps': [3, 4, 5],
        'batch_size': [8],  # Keep constant
    }
    
    # Generate all combinations
    keys = search_space.keys()
    values = search_space.values()
    configs = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"\n🔬 Grid Search Space:")
    for key, vals in search_space.items():
        print(f"   {key:15s}: {vals}")
    print(f"\n📊 Total configurations to test: {len(configs)}")
    
    return configs


def random_search_configs(n=20):
    """
    Random search - more efficient than grid search
    
    Samples random configurations from ranges
    """
    import numpy as np
    
    configs = []
    for i in range(n):
        config = {
            'dropout': np.random.uniform(0.3, 0.6),
            'weight_decay': 10 ** np.random.uniform(-6, -4),  # Log scale
            'learning_rate': 10 ** np.random.uniform(-4, -3),  # Log scale
            'num_steps': np.random.choice([3, 4, 5, 6]),
            'batch_size': 8,
        }
        configs.append(config)
    
    print(f"\n🎲 Random Search:")
    print(f"   Configurations to test: {n}")
    
    return configs


def smart_search_configs():
    """
    Smart search based on what we know so far
    
    Problem: Val acc stuck at 49.73%
    Likely causes:
    1. Model too weak (can't learn)
    2. Learning rate too high (unstable)
    3. Initialization issue
    
    Let's test configurations that address this
    """
    configs = [
        # Baseline (what README uses)
        {
            'dropout': 0.3,
            'weight_decay': 1.3e-6,
            'learning_rate': 1e-4,
            'num_steps': 8,
            'batch_size': 8,
            'name': 'README_baseline'
        },
        
        # Slightly stronger regularization
        {
            'dropout': 0.35,
            'weight_decay': 5e-6,
            'learning_rate': 1e-4,
            'num_steps': 6,
            'batch_size': 8,
            'name': 'mild_regularization'
        },
        
        # Lower learning rate (might help stability)
        {
            'dropout': 0.3,
            'weight_decay': 1.3e-6,
            'learning_rate': 5e-5,
            'num_steps': 6,
            'batch_size': 8,
            'name': 'lower_lr'
        },
        
        # Simpler model
        {
            'dropout': 0.3,
            'weight_decay': 1e-6,
            'learning_rate': 2e-4,
            'num_steps': 4,
            'batch_size': 8,
            'name': 'simpler_model'
        },
        
        # Higher learning rate
        {
            'dropout': 0.4,
            'weight_decay': 1e-5,
            'learning_rate': 5e-4,
            'num_steps': 4,
            'batch_size': 8,
            'name': 'higher_lr'
        },
    ]
    
    print(f"\n🎯 Smart Search:")
    print(f"   Testing {len(configs)} targeted configurations")
    for i, cfg in enumerate(configs, 1):
        print(f"   {i}. {cfg.get('name', f'config_{i}')}")
    
    return configs


def run_experiment(config, train_loader, val_loader, device, 
                   max_epochs=30, logger=None):
    """
    Run a single experiment with given configuration
    
    Returns results dict with metrics
    """
    from balanced_training_config import BalancedDevignModel
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {config.get('name', 'unnamed')}")
    print(f"{'='*80}")
    print(f"Config: {config}")
    
    # Create model
    model = BalancedDevignModel(
        input_dim=205,
        output_dim=2,
        hidden_dim=200,
        num_steps=config['num_steps'],
        dropout=config['dropout']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Early stop checker
    early_stop = EarlyStopChecker(patience=5, min_epochs=10)
    
    # Track metrics
    train_accs = []
    val_accs = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch)
            target = batch.y.squeeze().long()
            loss = criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += batch.num_graphs
        
        train_acc = train_correct / train_total
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                target = batch.y.squeeze().long()
                
                loss = criterion(output, target)
                val_loss += loss.item() * batch.num_graphs
                
                pred = output.argmax(dim=1)
                val_correct += (pred == target).sum().item()
                val_total += batch.num_graphs
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        val_loss = val_loss / val_total
        val_losses.append(val_acc)  # Track acc for stuck detection
        
        # Print progress
        gap = train_acc - val_acc
        print(f"Epoch {epoch+1:2d}: Train {train_acc:.2%}, "
              f"Val {val_acc:.2%}, Gap {gap:+.2%}, "
              f"Loss {val_loss:.4f}")
        
        # Check for early stopping
        should_stop, reason = early_stop.should_stop(
            epoch, train_acc, val_acc, val_losses
        )
        if should_stop:
            print(f"\n⚠️ Early stop: {reason}")
            break
    
    elapsed = time.time() - start_time
    
    # Collect results
    results = {
        'best_val_acc': max(val_accs),
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
        'final_gap': train_accs[-1] - val_accs[-1],
        'converged_epoch': val_accs.index(max(val_accs)) + 1,
        'total_epochs': len(train_accs),
        'time_seconds': elapsed,
        'stopped_early': len(train_accs) < max_epochs,
        'stop_reason': reason if should_stop else None,
    }
    
    print(f"\n📊 Results:")
    print(f"   Best Val Acc: {results['best_val_acc']:.2%}")
    print(f"   Final Train/Val: {results['final_train_acc']:.2%} / {results['final_val_acc']:.2%}")
    print(f"   Gap: {results['final_gap']:+.2%}")
    print(f"   Converged at epoch: {results['converged_epoch']}")
    
    # Log experiment
    if logger:
        logger.log_experiment(config, results)
    
    return results


def main():
    """
    Main hyperparameter search workflow
    """
    print("\n" + "#"*80)
    print("# AUTOMATED HYPERPARAMETER SEARCH")
    print("#"*80)
    
    # Initialize logger
    logger = ExperimentLogger('experiments_log.json')
    
    # Choose search strategy
    print("\n🔬 Search Strategy Options:")
    print("   1. Smart Search (5 targeted configs) - RECOMMENDED")
    print("   2. Random Search (20 random configs)")
    print("   3. Grid Search (144 configs - SLOW!)")
    
    choice = input("\nChoose strategy (1/2/3) [1]: ").strip() or "1"
    
    if choice == "1":
        configs = smart_search_configs()
    elif choice == "2":
        configs = random_search_configs(n=20)
    else:
        configs = grid_search_configs()
    
    print(f"\n🚀 Starting search with {len(configs)} configurations...")
    print(f"   Estimated time: {len(configs) * 5} minutes")
    print(f"   Results will be logged to: experiments_log.json")
    
    input("\nPress ENTER to start...")
    
    # NOTE: You need to provide train_loader and val_loader
    # This is a template - integrate with your actual data loading code
    
    print("\n⚠️ ERROR: This is a template script!")
    print("To use this, you need to:")
    print("1. Load your train_loader and val_loader")
    print("2. Call run_experiment() for each config")
    print("3. Analyze results with logger.print_summary()")
    
    print("\n📝 Integration code:")
    print("""
# In your main training script:
from auto_hyperparameter_search import (
    ExperimentLogger, run_experiment, smart_search_configs
)

# Load your data
train_loader, val_loader, _ = get_your_dataloaders()

# Initialize
logger = ExperimentLogger()
configs = smart_search_configs()

# Run experiments
for config in configs:
    results = run_experiment(
        config, train_loader, val_loader, 
        device='cuda', max_epochs=30, logger=logger
    )

# Analyze
logger.print_summary()
best_configs = logger.get_best_experiments(n=3)
print("\\nTop 3 configurations:")
for i, exp in enumerate(best_configs, 1):
    print(f"{i}. Val Acc: {exp['results']['best_val_acc']:.2%}")
    print(f"   Config: {exp['config']}")
""")


if __name__ == "__main__":
    main()