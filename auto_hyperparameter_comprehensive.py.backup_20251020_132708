# """
# Comprehensive Auto-Hyperparameter Tuning for CypherSec-AI
# Fixes all issues and systematically tests configurations

# CRITICAL FIXES IMPLEMENTED:
# 1. Gradient clipping
# 2. Class weights
# 3. Learning rate scheduling
# 4. Enhanced monitoring
# 5. Model collapse detection

# Run: python auto_hyperparameter_comprehensive.py
# Stop: Ctrl+C (progress is saved)
# """

# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# import pandas as pd
# import numpy as np
# import json
# import time
# from datetime import datetime
# from pathlib import Path
# import sys
# import traceback
# from collections import defaultdict
# import warnings
# warnings.filterwarnings('ignore')

# # Import your existing modules
# # Note: We'll use the adapter for data loading and model creation
# try:
#     import adapter  # Our working adapter
#     print("✓ Adapter imported successfully")
# except ImportError as e:
#     print(f"❌ Warning: {e}")
#     print("Make sure adapter.py is in the project root")

# class ComprehensiveHyperparameterSearch:
#     """
#     Exhaustive hyperparameter search with all fixes applied
#     """
    
#     def __init__(self, 
#                  data_path='data/input',
#                  log_file='hyperparameter_search_log.txt',
#                  results_file='hyperparameter_results.json',
#                  checkpoint_file='search_checkpoint.json'):
        
#         self.data_path = data_path
#         self.log_file = log_file
#         self.results_file = results_file
#         self.checkpoint_file = checkpoint_file
        
#         self.results = []
#         self.best_config = None
#         self.best_score = 0
#         self.trial_count = 0
        
#         # Device
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load data once
#         print("Loading datasets...")
#         self.train_loader, self.val_loader, self.test_loader = self._load_data()
        
#         # Calculate class weights once
#         self.class_weights = self._calculate_class_weights()
        
#         # Initialize log file
#         self._init_log_file()
    
#     def _init_log_file(self):
#         """Initialize log file with header"""
#         with open(self.log_file, 'w', encoding='utf-8') as f:
#             f.write("="*80 + "\n")
#             f.write("COMPREHENSIVE HYPERPARAMETER SEARCH LOG\n")
#             f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#             f.write(f"Device: {self.device}\n")
#             f.write(f"Class Weights: {self.class_weights}\n")
#             f.write("="*80 + "\n\n")
    
#     def _log(self, message, print_too=True):
#         """Log message to file and optionally print"""
#         timestamp = datetime.now().strftime('%H:%M:%S')
#         log_msg = f"[{timestamp}] {message}"
        
#         with open(self.log_file, 'a', encoding='utf-8') as f:
#             f.write(log_msg + "\n")
        
#         if print_too:
#             print(log_msg)
    
#     def _load_data(self):
#         """Load train/val/test datasets using our working adapter"""
#         try:
#             # Use the working adapter we just created
#             from adapter import get_data_loaders
            
#             print("Loading data using working adapter...")
#             train_loader, val_loader, test_loader = get_data_loaders(batch_size=8)
            
#             self._log(f"✓ Data loaded - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
            
#             return train_loader, val_loader, test_loader
            
#         except Exception as e:
#             self._log(f"❌ Data loading failed: {e}")
#             raise
    
#     def _load_dataset(self, split):
#         """Load dataset for given split"""
#         # Implement based on your data structure
#         # This is a placeholder - adapt to your actual data loading
#         pass
    
#     def _calculate_class_weights(self):
#         """Calculate class weights from training data"""
#         try:
#             all_labels = []
#             for batch in self.train_loader:
#                 if hasattr(batch, 'y'):
#                     all_labels.extend(batch.y.cpu().numpy())
#                 elif isinstance(batch, tuple):
#                     all_labels.extend(batch[1].cpu().numpy())
            
#             all_labels = np.array(all_labels)
#             unique, counts = np.unique(all_labels, return_counts=True)
            
#             total = len(all_labels)
#             weights = torch.zeros(2)
            
#             for cls, count in zip(unique, counts):
#                 weights[int(cls)] = total / (2 * count)
            
#             self._log(f"Calculated class weights: {weights}")
#             return weights.to(self.device)
            
#         except Exception as e:
#             self._log(f"Could not calculate class weights: {e}")
#             return torch.ones(2).to(self.device)
    
#     def define_search_space(self):
#         """
#         Define comprehensive search space
#         """
#         search_space = {
#             # Critical parameters
#             'learning_rate': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
#             'gradient_clip': [0.5, 1.0, 2.0, 5.0],
#             'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#             'weight_decay': [0, 1e-6, 1e-5, 5e-5, 1e-4],
            
#             # Model architecture
#             'hidden_dim': [64, 128, 192, 256],
#             'num_layers': [2, 3, 4, 5],
            
#             # Training
#             'batch_size': [4, 8, 16, 32],
#             'optimizer': ['adam', 'adamw'],
            
#             # Class weight multiplier (to handle extreme imbalance)
#             'class_weight_scale': [0.5, 1.0, 1.5, 2.0],
            
#             # Scheduler
#             'scheduler': ['plateau', 'step', 'none'],
#             'scheduler_patience': [3, 5, 7],
#             'scheduler_factor': [0.3, 0.5, 0.7]
#         }
        
#         return search_space
    
#     def generate_configs(self, search_space, strategy='smart'):
#         """
#         Generate configurations to test
        
#         Args:
#             strategy: 'smart', 'random', or 'grid'
#         """
#         if strategy == 'smart':
#             return self._generate_smart_configs(search_space)
#         elif strategy == 'random':
#             return self._generate_random_configs(search_space)
#         elif strategy == 'grid':
#             return self._generate_grid_configs(search_space)
    
#     def _generate_smart_configs(self, search_space):
#         """
#         Generate smart configurations based on known good practices
#         """
#         import random
        
#         # Priority configurations (from literature and experience)
#         priority_configs = [
#             # Original Devign paper config (with fixes)
#             {
#                 'learning_rate': 1e-4,
#                 'gradient_clip': 1.0,
#                 'dropout': 0.3,
#                 'weight_decay': 1.3e-6,
#                 'hidden_dim': 128,
#                 'num_layers': 3,
#                 'batch_size': 8,
#                 'optimizer': 'adam',
#                 'class_weight_scale': 1.0,
#                 'scheduler': 'plateau',
#                 'scheduler_patience': 5,
#                 'scheduler_factor': 0.5
#             },
#             # Conservative (less prone to overfitting)
#             {
#                 'learning_rate': 5e-5,
#                 'gradient_clip': 1.0,
#                 'dropout': 0.5,
#                 'weight_decay': 1e-4,
#                 'hidden_dim': 128,
#                 'num_layers': 2,
#                 'batch_size': 16,
#                 'optimizer': 'adamw',
#                 'class_weight_scale': 1.5,
#                 'scheduler': 'plateau',
#                 'scheduler_patience': 5,
#                 'scheduler_factor': 0.5
#             },
#             # Aggressive (faster learning)
#             {
#                 'learning_rate': 2e-4,
#                 'gradient_clip': 2.0,
#                 'dropout': 0.2,
#                 'weight_decay': 1e-5,
#                 'hidden_dim': 256,
#                 'num_layers': 3,
#                 'batch_size': 8,
#                 'optimizer': 'adam',
#                 'class_weight_scale': 2.0,
#                 'scheduler': 'step',
#                 'scheduler_patience': 5,
#                 'scheduler_factor': 0.7
#             },
#             # Deep model
#             {
#                 'learning_rate': 5e-5,
#                 'gradient_clip': 1.0,
#                 'dropout': 0.4,
#                 'weight_decay': 5e-5,
#                 'hidden_dim': 192,
#                 'num_layers': 4,
#                 'batch_size': 8,
#                 'optimizer': 'adamw',
#                 'class_weight_scale': 1.5,
#                 'scheduler': 'plateau',
#                 'scheduler_patience': 7,
#                 'scheduler_factor': 0.5
#             },
#             # Wide model
#             {
#                 'learning_rate': 1e-4,
#                 'gradient_clip': 1.0,
#                 'dropout': 0.3,
#                 'weight_decay': 1e-5,
#                 'hidden_dim': 256,
#                 'num_layers': 2,
#                 'batch_size': 16,
#                 'optimizer': 'adam',
#                 'class_weight_scale': 1.0,
#                 'scheduler': 'plateau',
#                 'scheduler_patience': 5,
#                 'scheduler_factor': 0.5
#             }
#         ]
        
#         # Generate infinite random configs for exploration
#         while True:
#             # Return priority configs first
#             for config in priority_configs:
#                 yield config
            
#             # Then generate random configs
#             while True:
#                 config = {
#                     param: random.choice(values)
#                     for param, values in search_space.items()
#                 }
#                 yield config
    
#     def create_model(self, config):
#         """Create model with given configuration using our working adapter"""
#         try:
#             # Use the working adapter we just created
#             from adapter import create_model
            
#             model = create_model(config)
            
#             # The adapter already handles device placement
#             self._log(f"✓ Model created with config: {config}")
#             return model
            
#         except Exception as e:
#             self._log(f"❌ Error creating model: {e}")
#             raise
    
#     def _get_input_dim(self):
#         """Get input dimension from data"""
#         # Get from first batch
#         for batch in self.train_loader:
#             if hasattr(batch, 'x'):
#                 return batch.x.size(1)
#             break
#         return 128  # Default
    
#     def train_epoch(self, model, optimizer, criterion, config):
#         """Train for one epoch with all fixes"""
#         model.train()
        
#         total_loss = 0
#         all_predictions = []
#         all_targets = []
#         gradient_norms = []
        
#         for batch_idx, batch in enumerate(self.train_loader):
#             batch = batch.to(self.device)
            
#             # Forward pass
#             optimizer.zero_grad()
#             outputs = model(batch)
            
#             # Get targets
#             if hasattr(batch, 'y'):
#                 targets = batch.y
#             else:
#                 targets = batch[1]
            
#             # Compute loss
#             loss = criterion(outputs, targets)
            
#             # Backward pass
#             loss.backward()
            
#             # CRITICAL: Gradient clipping
#             grad_norm = torch.nn.utils.clip_grad_norm_(
#                 model.parameters(),
#                 max_norm=config['gradient_clip']
#             )
#             gradient_norms.append(grad_norm.item())
            
#             # Optimizer step
#             optimizer.step()
            
#             # Track metrics
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             all_predictions.extend(predicted.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())
        
#         # Calculate metrics
#         avg_loss = total_loss / len(self.train_loader)
#         accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
#         # Gradient statistics
#         grad_stats = {
#             'mean': np.mean(gradient_norms),
#             'max': np.max(gradient_norms),
#             'min': np.min(gradient_norms),
#             'std': np.std(gradient_norms)
#         }
        
#         # Per-class accuracy
#         class_acc = {}
#         for cls in [0, 1]:
#             mask = np.array(all_targets) == cls
#             if mask.sum() > 0:
#                 class_acc[cls] = np.mean(
#                     np.array(all_predictions)[mask] == np.array(all_targets)[mask]
#                 )
        
#         return {
#             'loss': avg_loss,
#             'accuracy': accuracy,
#             'class_accuracy': class_acc,
#             'gradient_stats': grad_stats,
#             'unique_predictions': len(set(all_predictions))
#         }
    
#     def validate_epoch(self, model, criterion):
#         """Validate model"""
#         model.eval()
        
#         total_loss = 0
#         all_predictions = []
#         all_targets = []
        
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 batch = batch.to(self.device)
                
#                 outputs = model(batch)
                
#                 if hasattr(batch, 'y'):
#                     targets = batch.y
#                 else:
#                     targets = batch[1]
                
#                 loss = criterion(outputs, targets)
#                 total_loss += loss.item()
                
#                 _, predicted = torch.max(outputs, 1)
#                 all_predictions.extend(predicted.cpu().numpy())
#                 all_targets.extend(targets.cpu().numpy())
        
#         avg_loss = total_loss / len(self.val_loader)
#         accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
#         # Per-class accuracy
#         class_acc = {}
#         for cls in [0, 1]:
#             mask = np.array(all_targets) == cls
#             if mask.sum() > 0:
#                 class_acc[cls] = np.mean(
#                     np.array(all_predictions)[mask] == np.array(all_targets)[mask]
#                 )
        
#         # Check for model collapse
#         unique_preds = len(set(all_predictions))
        
#         return {
#             'loss': avg_loss,
#             'accuracy': accuracy,
#             'class_accuracy': class_acc,
#             'unique_predictions': unique_preds,
#             'predictions': all_predictions,
#             'targets': all_targets
#         }
    
#     def train_with_config(self, config, max_epochs=30):
#         """
#         Train model with given configuration
#         """
#         self.trial_count += 1
#         trial_id = self.trial_count
        
#         self._log("="*80)
#         self._log(f"TRIAL {trial_id} - Starting")
#         self._log("="*80)
#         self._log(f"Configuration: {json.dumps(config, indent=2)}")
        
#         try:
#             # Create model
#             model = self.create_model(config)
            
#             # Setup optimizer
#             if config['optimizer'] == 'adam':
#                 optimizer = torch.optim.Adam(
#                     model.parameters(),
#                     lr=config['learning_rate'],
#                     weight_decay=config['weight_decay']
#                 )
#             else:  # adamw
#                 optimizer = torch.optim.AdamW(
#                     model.parameters(),
#                     lr=config['learning_rate'],
#                     weight_decay=config['weight_decay']
#                 )
            
#             # Setup loss function with scaled class weights
#             scaled_weights = self.class_weights * config['class_weight_scale']
#             criterion = nn.CrossEntropyLoss(weight=scaled_weights)
            
#             # Setup scheduler
#             scheduler = None
#             if config['scheduler'] == 'plateau':
#                 scheduler = ReduceLROnPlateau(
#                     optimizer,
#                     mode='min',
#                     factor=config['scheduler_factor'],
#                     patience=config['scheduler_patience']
#                 )
#             elif config['scheduler'] == 'step':
#                 scheduler = StepLR(
#                     optimizer,
#                     step_size=config['scheduler_patience'],
#                     gamma=config['scheduler_factor']
#                 )
            
#             # Training loop
#             best_val_loss = float('inf')
#             best_val_acc = 0
#             patience_counter = 0
#             patience_limit = 10
            
#             history = {
#                 'train_loss': [],
#                 'train_acc': [],
#                 'val_loss': [],
#                 'val_acc': []
#             }
            
#             for epoch in range(1, max_epochs + 1):
#                 # Train
#                 train_metrics = self.train_epoch(model, optimizer, criterion, config)
                
#                 # Validate
#                 val_metrics = self.validate_epoch(model, criterion)
                
#                 # Update scheduler
#                 if scheduler:
#                     if config['scheduler'] == 'plateau':
#                         scheduler.step(val_metrics['loss'])
#                     else:
#                         scheduler.step()
                
#                 # Store history
#                 history['train_loss'].append(train_metrics['loss'])
#                 history['train_acc'].append(train_metrics['accuracy'])
#                 history['val_loss'].append(val_metrics['loss'])
#                 history['val_acc'].append(val_metrics['accuracy'])
                
#                 # Log epoch
#                 log_msg = f"Trial {trial_id} - Epoch {epoch}/{max_epochs}\n"
#                 log_msg += f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}"
#                 log_msg += f" [C0: {train_metrics['class_accuracy'].get(0, 0):.3f}, C1: {train_metrics['class_accuracy'].get(1, 0):.3f}]\n"
#                 log_msg += f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}"
#                 log_msg += f" [C0: {val_metrics['class_accuracy'].get(0, 0):.3f}, C1: {val_metrics['class_accuracy'].get(1, 0):.3f}]\n"
#                 log_msg += f"  Unique preds: {val_metrics['unique_predictions']}, "
#                 log_msg += f"Grad max: {train_metrics['gradient_stats']['max']:.2f}"
                
#                 self._log(log_msg, print_too=(epoch % 5 == 0 or epoch <= 3))
                
#                 # Check for model collapse
#                 if val_metrics['unique_predictions'] == 1:
#                     self._log(f"  ❌ MODEL COLLAPSED - Predicting only 1 class!")
#                     break
                
#                 # Track best
#                 if val_metrics['accuracy'] > best_val_acc:
#                     best_val_acc = val_metrics['accuracy']
#                     best_val_loss = val_metrics['loss']
#                     patience_counter = 0
#                 else:
#                     patience_counter += 1
                
#                 # Early stopping
#                 if patience_counter >= patience_limit:
#                     self._log(f"  Early stopping at epoch {epoch}")
#                     break
                
#                 # Check for gradient explosion
#                 if train_metrics['gradient_stats']['max'] > 20:
#                     self._log(f"  ⚠️  Gradient explosion detected! Max: {train_metrics['gradient_stats']['max']:.2f}")
            
#             # Final evaluation
#             final_results = {
#                 'trial_id': trial_id,
#                 'config': config,
#                 'best_val_accuracy': best_val_acc,
#                 'best_val_loss': best_val_loss,
#                 'final_val_accuracy': history['val_acc'][-1],
#                 'final_train_accuracy': history['train_acc'][-1],
#                 'epochs_trained': len(history['train_loss']),
#                 'history': history,
#                 'timestamp': datetime.now().isoformat()
#             }
            
#             # Determine status
#             if val_metrics['unique_predictions'] == 1:
#                 status = "COLLAPSED"
#             elif best_val_acc < 0.52:
#                 status = "POOR"
#             elif abs(history['train_acc'][-1] - best_val_acc) > 0.3:
#                 status = "OVERFIT"
#             elif best_val_acc > 0.65:
#                 status = "EXCELLENT"
#             elif best_val_acc > 0.60:
#                 status = "GOOD"
#             else:
#                 status = "ACCEPTABLE"
            
#             final_results['status'] = status
            
#             # Log summary
#             self._log("\n" + "-"*80)
#             self._log(f"Trial {trial_id} Complete - Status: {status}")
#             self._log(f"  Best Val Acc: {best_val_acc:.4f}")
#             self._log(f"  Final Val Acc: {history['val_acc'][-1]:.4f}")
#             self._log(f"  Train-Val Gap: {abs(history['train_acc'][-1] - best_val_acc):.4f}")
#             self._log("-"*80 + "\n")
            
#             # Update best
#             if best_val_acc > self.best_score:
#                 self.best_score = best_val_acc
#                 self.best_config = config
#                 self._log(f"🎉 NEW BEST CONFIG! Val Acc: {best_val_acc:.4f}")
            
#             return final_results
            
#         except Exception as e:
#             self._log(f"❌ Trial {trial_id} failed with error: {str(e)}")
#             self._log(traceback.format_exc())
#             return {
#                 'trial_id': trial_id,
#                 'config': config,
#                 'status': 'FAILED',
#                 'error': str(e),
#                 'timestamp': datetime.now().isoformat()
#             }
    
#     def save_results(self):
#         """Save results to JSON file"""
#         output = {
#             'total_trials': self.trial_count,
#             'best_config': self.best_config,
#             'best_score': self.best_score,
#             'all_results': self.results,
#             'last_updated': datetime.now().isoformat()
#         }
        
#         with open(self.results_file, 'w') as f:
#             json.dump(output, f, indent=2)
        
#         self._log(f"✓ Results saved to {self.results_file}")
    
#     def save_checkpoint(self):
#         """Save checkpoint for resuming"""
#         checkpoint = {
#             'trial_count': self.trial_count,
#             'best_config': self.best_config,
#             'best_score': self.best_score,
#             'timestamp': datetime.now().isoformat()
#         }
        
#         with open(self.checkpoint_file, 'w') as f:
#             json.dump(checkpoint, f, indent=2)
    
#     def run_search(self, max_trials=None):
#         """
#         Run hyperparameter search
        
#         Args:
#             max_trials: Maximum number of trials (None = infinite)
#         """
#         self._log("="*80)
#         self._log("STARTING COMPREHENSIVE HYPERPARAMETER SEARCH")
#         self._log("="*80)
#         self._log(f"Max trials: {'Infinite (Ctrl+C to stop)' if max_trials is None else max_trials}")
#         self._log(f"Device: {self.device}")
#         self._log(f"Results will be saved to: {self.results_file}")
#         self._log(f"Log file: {self.log_file}")
#         self._log("="*80 + "\n")
        
#         # Define search space
#         search_space = self.define_search_space()
        
#         # Generate configurations
#         config_generator = self.generate_configs(search_space, strategy='smart')
        
#         try:
#             trial_num = 0
#             for config in config_generator:
#                 trial_num += 1
                
#                 # Check if we've hit max trials
#                 if max_trials and trial_num > max_trials:
#                     break
                
#                 # Train with this configuration
#                 results = self.train_with_config(config, max_epochs=30)
                
#                 # Store results
#                 self.results.append(results)
                
#                 # Save progress
#                 self.save_results()
#                 self.save_checkpoint()
                
#                 # Print summary every 5 trials
#                 if trial_num % 5 == 0:
#                     self._log("\n" + "="*80)
#                     self._log(f"PROGRESS UPDATE - Completed {trial_num} trials")
#                     self._log(f"Best so far: {self.best_score:.4f}")
#                     if self.best_config:
#                         self._log(f"Best config: LR={self.best_config['learning_rate']}, "
#                                  f"Dropout={self.best_config['dropout']}, "
#                                  f"Clip={self.best_config['gradient_clip']}")
#                     self._log("="*80 + "\n")
        
#         except KeyboardInterrupt:
#             self._log("\n\n" + "="*80)
#             self._log("SEARCH INTERRUPTED BY USER")
#             self._log("="*80)
        
#         finally:
#             # Final save
#             self.save_results()
            
#             # Print final summary
#             self._log("\n\n" + "="*80)
#             self._log("SEARCH COMPLETE")
#             self._log("="*80)
#             self._log(f"Total trials: {self.trial_count}")
#             self._log(f"Best validation accuracy: {self.best_score:.4f}")
#             self._log(f"\nBest configuration:")
#             self._log(json.dumps(self.best_config, indent=2))
#             self._log("="*80)
            
#             # Print top 5 configs
#             sorted_results = sorted(
#                 [r for r in self.results if 'best_val_accuracy' in r],
#                 key=lambda x: x['best_val_accuracy'],
#                 reverse=True
#             )[:5]
            
#             self._log("\n\nTOP 5 CONFIGURATIONS:")
#             self._log("="*80)
#             for i, result in enumerate(sorted_results, 1):
#                 self._log(f"\n{i}. Val Acc: {result['best_val_accuracy']:.4f} - Status: {result['status']}")
#                 self._log(f"   Config: {json.dumps(result['config'], indent=6)}")
            
#             self._log("\n" + "="*80)
#             self._log(f"Full results saved to: {self.results_file}")
#             self._log(f"Full log saved to: {self.log_file}")
#             self._log("="*80)

# def main():
#     """Main execution"""
#     print("\n" + "="*80)
#     print("CypherSec-AI - Comprehensive Hyperparameter Search")
#     print("="*80)
#     print("\nThis will:")
#     print("  1. Test multiple configurations systematically")
#     print("  2. Apply ALL critical fixes (gradient clipping, class weights, etc.)")
#     print("  3. Log everything to hyperparameter_search_log.txt")
#     print("  4. Save results to hyperparameter_results.json")
#     print("  5. Run until stopped (Ctrl+C)")
#     print("\nPress Ctrl+C at any time to stop (progress will be saved)")
#     print("="*80 + "\n")
    
#     # Confirm
#     response = input("Continue? (yes/no): ")
#     if response.lower() not in ['yes', 'y']:
#         print("Cancelled.")
#         return
    
#     # Create searcher
#     searcher = ComprehensiveHyperparameterSearch()
    
#     # Run search
#     searcher.run_search(max_trials=None)  # Infinite trials

# if __name__ == "__main__":
#     main()





"""
Comprehensive Auto-Hyperparameter Tuning for CypherSec-AI
Fixes all issues and systematically tests configurations

CRITICAL FIXES IMPLEMENTED:
1. Gradient clipping
2. Class weights
3. Learning rate scheduling
4. Enhanced monitoring
5. Model collapse detection

Run: python auto_hyperparameter_comprehensive.py
Stop: Ctrl+C (progress is saved)
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import traceback
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules
try:
    from src.process.devign import DevignModel
    from src.data.datamanger import DataManager
    from torch_geometric.loader import DataLoader
except ImportError as e:
    print(f"Warning: {e}")
    print("Make sure you're running from project root")

class ComprehensiveHyperparameterSearch:
    """
    Exhaustive hyperparameter search with all fixes applied
    """
    
    def __init__(self, 
                 data_path='data/input',
                 log_file='hyperparameter_search_log.txt',
                 results_file='hyperparameter_results.json',
                 checkpoint_file='search_checkpoint.json'):
        
        self.data_path = data_path
        self.log_file = log_file
        self.results_file = results_file
        self.checkpoint_file = checkpoint_file
        
        self.results = []
        self.best_config = None
        self.best_score = 0
        self.trial_count = 0
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data once
        print("Loading datasets...")
        self.train_loader, self.val_loader, self.test_loader = self._load_data()
        
        # Calculate class weights once
        self.class_weights = self._calculate_class_weights()
        
        # Initialize log file
        self._init_log_file()
    
    def _init_log_file(self):
        """Initialize log file with header"""
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE HYPERPARAMETER SEARCH LOG\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Class Weights: {self.class_weights}\n")
            f.write("="*80 + "\n\n")
    
    def _log(self, message, print_too=True):
        """Log message to file and optionally print"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")
        
        if print_too:
            print(log_msg)
    
    def _load_data(self):
        """Load train/val/test datasets"""
        try:
            # Load using your DataManager
            dm = DataManager()
            train_loader = dm.get_train_loader(batch_size=8)
            val_loader = dm.get_val_loader(batch_size=8)
            test_loader = dm.get_test_loader(batch_size=8)
            return train_loader, val_loader, test_loader
        except:
            # Fallback: Load manually
            print("Using fallback data loading...")
            from torch_geometric.data import Dataset, DataLoader
            
            # You'll need to implement this based on your data structure
            # This is a placeholder
            train_dataset = self._load_dataset('train')
            val_dataset = self._load_dataset('val')
            test_dataset = self._load_dataset('test')
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
            
            return train_loader, val_loader, test_loader
    
    def _load_dataset(self, split):
        """Load dataset for given split"""
        # Implement based on your data structure
        # This is a placeholder - adapt to your actual data loading
        pass
    
    def _calculate_class_weights(self):
        """Calculate class weights from training data"""
        try:
            all_labels = []
            for batch in self.train_loader:
                if hasattr(batch, 'y'):
                    all_labels.extend(batch.y.cpu().numpy())
                elif isinstance(batch, tuple):
                    all_labels.extend(batch[1].cpu().numpy())
            
            all_labels = np.array(all_labels)
            unique, counts = np.unique(all_labels, return_counts=True)
            
            total = len(all_labels)
            weights = torch.zeros(2)
            
            for cls, count in zip(unique, counts):
                weights[int(cls)] = total / (2 * count)
            
            self._log(f"Calculated class weights: {weights}")
            return weights.to(self.device)
            
        except Exception as e:
            self._log(f"Could not calculate class weights: {e}")
            return torch.ones(2).to(self.device)
    
    def define_search_space(self):
        """
        Define comprehensive search space
        """
        search_space = {
            # Critical parameters
            'learning_rate': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
            'gradient_clip': [0.5, 1.0, 2.0, 5.0],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'weight_decay': [0, 1e-6, 1e-5, 5e-5, 1e-4],
            
            # Model architecture
            'hidden_dim': [64, 128, 192, 256],
            'num_layers': [2, 3, 4, 5],
            
            # Training
            'batch_size': [4, 8, 16, 32],
            'optimizer': ['adam', 'adamw'],
            
            # Class weight multiplier (to handle extreme imbalance)
            'class_weight_scale': [0.5, 1.0, 1.5, 2.0],
            
            # Scheduler
            'scheduler': ['plateau', 'step', 'none'],
            'scheduler_patience': [3, 5, 7],
            'scheduler_factor': [0.3, 0.5, 0.7]
        }
        
        return search_space
    
    def generate_configs(self, search_space, strategy='smart'):
        """
        Generate configurations to test
        
        Args:
            strategy: 'smart', 'random', or 'grid'
        """
        if strategy == 'smart':
            return self._generate_smart_configs(search_space)
        elif strategy == 'random':
            return self._generate_random_configs(search_space)
        elif strategy == 'grid':
            return self._generate_grid_configs(search_space)
    
    def _generate_smart_configs(self, search_space):
        """
        Generate smart configurations based on known good practices
        """
        import random
        
        # Priority configurations (from literature and experience)
        priority_configs = [
            # Original Devign paper config (with fixes)
            {
                'learning_rate': 1e-4,
                'gradient_clip': 1.0,
                'dropout': 0.3,
                'weight_decay': 1.3e-6,
                'hidden_dim': 128,
                'num_layers': 3,
                'batch_size': 8,
                'optimizer': 'adam',
                'class_weight_scale': 1.0,
                'scheduler': 'plateau',
                'scheduler_patience': 5,
                'scheduler_factor': 0.5
            },
            # Conservative (less prone to overfitting)
            {
                'learning_rate': 5e-5,
                'gradient_clip': 1.0,
                'dropout': 0.5,
                'weight_decay': 1e-4,
                'hidden_dim': 128,
                'num_layers': 2,
                'batch_size': 16,
                'optimizer': 'adamw',
                'class_weight_scale': 1.5,
                'scheduler': 'plateau',
                'scheduler_patience': 5,
                'scheduler_factor': 0.5
            },
            # Aggressive (faster learning)
            {
                'learning_rate': 2e-4,
                'gradient_clip': 2.0,
                'dropout': 0.2,
                'weight_decay': 1e-5,
                'hidden_dim': 256,
                'num_layers': 3,
                'batch_size': 8,
                'optimizer': 'adam',
                'class_weight_scale': 2.0,
                'scheduler': 'step',
                'scheduler_patience': 5,
                'scheduler_factor': 0.7
            },
            # Deep model
            {
                'learning_rate': 5e-5,
                'gradient_clip': 1.0,
                'dropout': 0.4,
                'weight_decay': 5e-5,
                'hidden_dim': 192,
                'num_layers': 4,
                'batch_size': 8,
                'optimizer': 'adamw',
                'class_weight_scale': 1.5,
                'scheduler': 'plateau',
                'scheduler_patience': 7,
                'scheduler_factor': 0.5
            },
            # Wide model
            {
                'learning_rate': 1e-4,
                'gradient_clip': 1.0,
                'dropout': 0.3,
                'weight_decay': 1e-5,
                'hidden_dim': 256,
                'num_layers': 2,
                'batch_size': 16,
                'optimizer': 'adam',
                'class_weight_scale': 1.0,
                'scheduler': 'plateau',
                'scheduler_patience': 5,
                'scheduler_factor': 0.5
            }
        ]
        
        # Generate infinite random configs for exploration
        while True:
            # Return priority configs first
            for config in priority_configs:
                yield config
            
            # Then generate random configs
            while True:
                config = {
                    param: random.choice(values)
                    for param, values in search_space.items()
                }
                yield config
    
    def create_model(self, config):
        """Create model with given configuration"""
        try:
            # Load your model - adapt this to your actual model initialization
            model = DevignModel(
                input_dim=self._get_input_dim(),
                hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'],
                dropout=config['dropout']
            )
            return model.to(self.device)
        except Exception as e:
            self._log(f"Error creating model: {e}")
            raise
    
    def _get_input_dim(self):
        """Get input dimension from data"""
        # Get from first batch
        for batch in self.train_loader:
            if hasattr(batch, 'x'):
                return batch.x.size(1)
            break
        return 128  # Default
    
    def train_epoch(self, model, optimizer, criterion, config):
        """Train for one epoch with all fixes"""
        model.train()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        gradient_norms = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            
            # Get targets
            if hasattr(batch, 'y'):
                targets = batch.y
            else:
                targets = batch[1]
            
            # CRITICAL FIX: Convert targets to Long type for CrossEntropyLoss
            targets = targets.long()
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # CRITICAL: Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config['gradient_clip']
            )
            gradient_norms.append(grad_norm.item())
            
            # Optimizer step
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        # Gradient statistics
        grad_stats = {
            'mean': np.mean(gradient_norms),
            'max': np.max(gradient_norms),
            'min': np.min(gradient_norms),
            'std': np.std(gradient_norms)
        }
        
        # Per-class accuracy
        class_acc = {}
        for cls in [0, 1]:
            mask = np.array(all_targets) == cls
            if mask.sum() > 0:
                class_acc[cls] = np.mean(
                    np.array(all_predictions)[mask] == np.array(all_targets)[mask]
                )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracy': class_acc,
            'gradient_stats': grad_stats,
            'unique_predictions': len(set(all_predictions))
        }
    
    def validate_epoch(self, model, criterion):
        """Validate model"""
        model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                outputs = model(batch)
                
                if hasattr(batch, 'y'):
                    targets = batch.y
                else:
                    targets = batch[1]
                
                # HOTFIX: Convert targets to Long type for CrossEntropyLoss
                targets = targets.long()
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        # Per-class accuracy
        class_acc = {}
        for cls in [0, 1]:
            mask = np.array(all_targets) == cls
            if mask.sum() > 0:
                class_acc[cls] = np.mean(
                    np.array(all_predictions)[mask] == np.array(all_targets)[mask]
                )
        
        # Check for model collapse
        unique_preds = len(set(all_predictions))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracy': class_acc,
            'unique_predictions': unique_preds,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train_with_config(self, config, max_epochs=30):
        """
        Train model with given configuration
        """
        self.trial_count += 1
        trial_id = self.trial_count
        
        self._log("="*80)
        self._log(f"TRIAL {trial_id} - Starting")
        self._log("="*80)
        self._log(f"Configuration: {json.dumps(config, indent=2)}")
        
        try:
            # Create model
            model = self.create_model(config)
            
            # Setup optimizer
            if config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            else:  # adamw
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
            
            # Setup loss function with scaled class weights
            scaled_weights = self.class_weights * config['class_weight_scale']
            criterion = nn.CrossEntropyLoss(weight=scaled_weights)
            
            # Setup scheduler
            scheduler = None
            if config['scheduler'] == 'plateau':
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=config['scheduler_factor'],
                    patience=config['scheduler_patience'],
                    verbose=False
                )
            elif config['scheduler'] == 'step':
                scheduler = StepLR(
                    optimizer,
                    step_size=config['scheduler_patience'],
                    gamma=config['scheduler_factor']
                )
            
            # Training loop
            best_val_loss = float('inf')
            best_val_acc = 0
            patience_counter = 0
            patience_limit = 10
            
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            
            for epoch in range(1, max_epochs + 1):
                # Train
                train_metrics = self.train_epoch(model, optimizer, criterion, config)
                
                # Validate
                val_metrics = self.validate_epoch(model, criterion)
                
                # Update scheduler
                if scheduler:
                    if config['scheduler'] == 'plateau':
                        scheduler.step(val_metrics['loss'])
                    else:
                        scheduler.step()
                
                # Store history
                history['train_loss'].append(train_metrics['loss'])
                history['train_acc'].append(train_metrics['accuracy'])
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                
                # Log epoch
                log_msg = f"Trial {trial_id} - Epoch {epoch}/{max_epochs}\n"
                log_msg += f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.4f}"
                log_msg += f" [C0: {train_metrics['class_accuracy'].get(0, 0):.3f}, C1: {train_metrics['class_accuracy'].get(1, 0):.3f}]\n"
                log_msg += f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.4f}"
                log_msg += f" [C0: {val_metrics['class_accuracy'].get(0, 0):.3f}, C1: {val_metrics['class_accuracy'].get(1, 0):.3f}]\n"
                log_msg += f"  Unique preds: {val_metrics['unique_predictions']}, "
                log_msg += f"Grad max: {train_metrics['gradient_stats']['max']:.2f}"
                
                self._log(log_msg, print_too=(epoch % 5 == 0 or epoch <= 3))
                
                # Check for model collapse
                if val_metrics['unique_predictions'] == 1:
                    self._log(f"  ❌ MODEL COLLAPSED - Predicting only 1 class!")
                    break
                
                # Track best
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience_limit:
                    self._log(f"  Early stopping at epoch {epoch}")
                    break
                
                # Check for gradient explosion
                if train_metrics['gradient_stats']['max'] > 20:
                    self._log(f"  ⚠️  Gradient explosion detected! Max: {train_metrics['gradient_stats']['max']:.2f}")
            
            # Final evaluation
            final_results = {
                'trial_id': trial_id,
                'config': config,
                'best_val_accuracy': best_val_acc,
                'best_val_loss': best_val_loss,
                'final_val_accuracy': history['val_acc'][-1],
                'final_train_accuracy': history['train_acc'][-1],
                'epochs_trained': len(history['train_loss']),
                'history': history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine status
            if val_metrics['unique_predictions'] == 1:
                status = "COLLAPSED"
            elif best_val_acc < 0.52:
                status = "POOR"
            elif abs(history['train_acc'][-1] - best_val_acc) > 0.3:
                status = "OVERFIT"
            elif best_val_acc > 0.65:
                status = "EXCELLENT"
            elif best_val_acc > 0.60:
                status = "GOOD"
            else:
                status = "ACCEPTABLE"
            
            final_results['status'] = status
            
            # Log summary
            self._log("\n" + "-"*80)
            self._log(f"Trial {trial_id} Complete - Status: {status}")
            self._log(f"  Best Val Acc: {best_val_acc:.4f}")
            self._log(f"  Final Val Acc: {history['val_acc'][-1]:.4f}")
            self._log(f"  Train-Val Gap: {abs(history['train_acc'][-1] - best_val_acc):.4f}")
            self._log("-"*80 + "\n")
            
            # Update best
            if best_val_acc > self.best_score:
                self.best_score = best_val_acc
                self.best_config = config
                self._log(f"🎉 NEW BEST CONFIG! Val Acc: {best_val_acc:.4f}")
            
            return final_results
            
        except Exception as e:
            self._log(f"❌ Trial {trial_id} failed with error: {str(e)}")
            self._log(traceback.format_exc())
            return {
                'trial_id': trial_id,
                'config': config,
                'status': 'FAILED',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def save_results(self):
        """Save results to JSON file"""
        output = {
            'total_trials': self.trial_count,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': self.results,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        self._log(f"✓ Results saved to {self.results_file}")
    
    def save_checkpoint(self):
        """Save checkpoint for resuming"""
        checkpoint = {
            'trial_count': self.trial_count,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def run_search(self, max_trials=None):
        """
        Run hyperparameter search
        
        Args:
            max_trials: Maximum number of trials (None = infinite)
        """
        self._log("="*80)
        self._log("STARTING COMPREHENSIVE HYPERPARAMETER SEARCH")
        self._log("="*80)
        self._log(f"Max trials: {'Infinite (Ctrl+C to stop)' if max_trials is None else max_trials}")
        self._log(f"Device: {self.device}")
        self._log(f"Results will be saved to: {self.results_file}")
        self._log(f"Log file: {self.log_file}")
        self._log("="*80 + "\n")
        
        # Define search space
        search_space = self.define_search_space()
        
        # Generate configurations
        config_generator = self.generate_configs(search_space, strategy='smart')
        
        try:
            trial_num = 0
            for config in config_generator:
                trial_num += 1
                
                # Check if we've hit max trials
                if max_trials and trial_num > max_trials:
                    break
                
                # Train with this configuration
                results = self.train_with_config(config, max_epochs=30)
                
                # Store results
                self.results.append(results)
                
                # Save progress
                self.save_results()
                self.save_checkpoint()
                
                # Print summary every 5 trials
                if trial_num % 5 == 0:
                    self._log("\n" + "="*80)
                    self._log(f"PROGRESS UPDATE - Completed {trial_num} trials")
                    self._log(f"Best so far: {self.best_score:.4f}")
                    if self.best_config:
                        self._log(f"Best config: LR={self.best_config['learning_rate']}, "
                                 f"Dropout={self.best_config['dropout']}, "
                                 f"Clip={self.best_config['gradient_clip']}")
                    self._log("="*80 + "\n")
        
        except KeyboardInterrupt:
            self._log("\n\n" + "="*80)
            self._log("SEARCH INTERRUPTED BY USER")
            self._log("="*80)
        
        finally:
            # Final save
            self.save_results()
            
            # Print final summary
            self._log("\n\n" + "="*80)
            self._log("SEARCH COMPLETE")
            self._log("="*80)
            self._log(f"Total trials: {self.trial_count}")
            self._log(f"Best validation accuracy: {self.best_score:.4f}")
            self._log(f"\nBest configuration:")
            self._log(json.dumps(self.best_config, indent=2))
            self._log("="*80)
            
            # Print top 5 configs
            sorted_results = sorted(
                [r for r in self.results if 'best_val_accuracy' in r],
                key=lambda x: x['best_val_accuracy'],
                reverse=True
            )[:5]
            
            self._log("\n\nTOP 5 CONFIGURATIONS:")
            self._log("="*80)
            for i, result in enumerate(sorted_results, 1):
                self._log(f"\n{i}. Val Acc: {result['best_val_accuracy']:.4f} - Status: {result['status']}")
                self._log(f"   Config: {json.dumps(result['config'], indent=6)}")
            
            self._log("\n" + "="*80)
            self._log(f"Full results saved to: {self.results_file}")
            self._log(f"Full log saved to: {self.log_file}")
            self._log("="*80)

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CypherSec-AI - Comprehensive Hyperparameter Search")
    print("="*80)
    print("\nThis will:")
    print("  1. Test multiple configurations systematically")
    print("  2. Apply ALL critical fixes (gradient clipping, class weights, etc.)")
    print("  3. Log everything to hyperparameter_search_log.txt")
    print("  4. Save results to hyperparameter_results.json")
    print("  5. Run until stopped (Ctrl+C)")
    print("\nPress Ctrl+C at any time to stop (progress will be saved)")
    print("="*80 + "\n")
    
    # Confirm
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Create searcher
    searcher = ComprehensiveHyperparameterSearch()
    
    # Run search
    searcher.run_search(max_trials=None)  # Infinite trials

if __name__ == "__main__":
    main()