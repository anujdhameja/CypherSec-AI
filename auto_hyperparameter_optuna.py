"""
Optuna-based Hyperparameter Optimization for Devign Model
Modern, feature-rich approach with pruning and visualization
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Import your existing modules
from src.data.datamanager import DataManager
from src.process.model import DevignModel
from src.process.modeling import Modeling
from src.utils.objects.input_dataset import InputDataset
from configs import configs

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("ERROR: optuna not installed!")
    print("Install with: pip install optuna")
    exit(1)


class OptunaHyperparameterSearch:
    def __init__(self, n_trials=30, study_name="devign_optimization"):
        """
        Initialize Optuna Hyperparameter Search
        
        Args:
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
        """
        self.n_trials = n_trials
        self.study_name = study_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results tracking
        self.results = {
            'trials': [],
            'best_config': None,
            'best_score': -float('inf'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print("\n" + "="*80)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        print(f"Using device: {self.device}")
        print(f"Study name: {study_name}")
        
        print("\n1. Loading data...")
        self._load_data()
        
        # Create Optuna study with median pruner
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximize validation accuracy
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(
                n_startup_trials=5,  # Don't prune first 5 trials
                n_warmup_steps=3     # Wait 3 epochs before pruning
            )
        )
        
    def _load_data(self):
        """Load and prepare datasets"""
        try:
            dataset_path = Path(configs['paths']['input'])
            
            # Load datasets
            train_ds = InputDataset(dataset_path / 'train.pkl')
            val_ds = InputDataset(dataset_path / 'val.pkl')
            
            print(f"   Loaded {len(train_ds)} training samples")
            print(f"   Loaded {len(val_ds)} validation samples")
            
            self.train_dataset = train_ds
            self.val_dataset = val_ds
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization
        
        Args:
            trial: Optuna trial object
            
        Returns:
            float: Validation accuracy to maximize
        """
        trial_num = trial.number + 1
        
        print("\n" + "="*80)
        print(f"TRIAL {trial_num}/{self.n_trials}")
        print("="*80)
        
        # Sample hyperparameters
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'num_steps': trial.suggest_int('num_steps', 2, 6),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 200, 256, 384]),
            'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
            # Additional parameters
            'loss_lambda': trial.suggest_float('loss_lambda', 1e-7, 1e-4, log=True),
        }
        
        print(f"Config: {json.dumps({k: f'{v:.2e}' if isinstance(v, float) and v < 0.01 else v for k, v in config.items()}, indent=2)}")
        
        start_time = time.time()
        
        try:
            # Update model configuration
            model_config = configs['devign']['model'].copy()
            model_config['hidden_size'] = config['hidden_dim']
            model_config['num_steps'] = config['num_steps']
            
            # Update conv layer dimensions
            model_config['conv_args']['conv1d_1']['in_channels'] = configs['data']['input_dataset']['nodes_dim']
            model_config['conv_args']['conv1d_2']['in_channels'] = config['hidden_dim']
            
            # Initialize model
            model = DevignModel(
                input_dim=configs['data']['input_dataset']['nodes_dim'],
                output_dim=2,
                model_params=model_config
            ).to(self.device)
            
            # Training configuration
            train_config = {
                'learning_rate': config['learning_rate'],
                'weight_decay': config['weight_decay'],
                'loss_lambda': config['loss_lambda'],
                'epochs': 15,
                'batch_size': config['batch_size'],
                'dropout': config['dropout']
            }
            
            # Initialize trainer
            modeling = Modeling(
                model=model,
                dataset={'train': self.train_dataset, 'val': self.val_dataset},
                params=train_config,
                device=self.device
            )
            
            # Train with pruning support
            best_val_acc = 0.0
            stuck_count = 0
            max_stuck = 3
            
            for epoch in range(train_config['epochs']):
                # Train one epoch
                train_acc, train_loss = modeling.train_epoch(epoch)
                val_acc, val_loss = modeling.validate_epoch(epoch)
                
                gap = train_acc - val_acc
                
                # Check if stuck
                is_stuck = abs(val_acc - 55.56) < 1.0 or abs(val_acc - 44.44) < 1.0
                stuck_indicator = "⚠️ STUCK" if is_stuck else "✓"
                
                if is_stuck:
                    stuck_count += 1
                else:
                    stuck_count = 0
                
                print(f"Epoch {epoch+1:2d}: Train {train_acc:.2f}%, "
                      f"Val {val_acc:.2f}%, Gap {gap:+.2f}%, "
                      f"Loss {train_loss:.4f} {stuck_indicator}")
                
                # Report intermediate value for pruning
                trial.report(val_acc, epoch)
                
                # Prune trial if not promising
                if trial.should_prune():
                    print("🔪 Trial pruned by Optuna")
                    raise optuna.TrialPruned()
                
                # Early stopping if stuck
                if stuck_count >= max_stuck:
                    print(f"Stopping early - model stuck for {max_stuck} epochs")
                    break
                
                # Track best
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
                'converged': not is_stuck,
                'pruned': False
            }
            
            self.results['trials'].append(trial_result)
            
            # Update best result
            if best_val_acc > self.results['best_score']:
                self.results['best_score'] = best_val_acc
                self.results['best_config'] = config
                print(f"   🎯 NEW BEST SCORE: {best_val_acc:.2f}%")
            
            # Save intermediate results
            self._save_results()
            
            return best_val_acc
            
        except optuna.TrialPruned:
            # Record pruned trial
            trial_result = {
                'trial_num': trial_num,
                'config': config,
                'pruned': True,
                'time_seconds': time.time() - start_time
            }
            self.results['trials'].append(trial_result)
            raise
            
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
            
            return 0.0
    
    def optimize(self):
        """Run Optuna optimization"""
        print(f"\n2. Starting Optuna Optimization ({self.n_trials} trials)...")
        print(f"   Early stopping: Unpromising trials will be pruned")
        print(f"   Estimated time: {self.n_trials * 1.5:.0f}-{self.n_trials * 3:.0f} minutes")
        
        try:
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                show_progress_bar=False,
                catch=(Exception,)
            )
            
            # Store final results
            self.results['optimization_summary'] = {
                'best_score': self.study.best_value,
                'best_params': self.study.best_params,
                'n_trials': len(self.study.trials),
                'n_complete_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_pruned_trials': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'total_time': sum(t['time_seconds'] for t in self.results['trials'] if 'time_seconds' in t)
            }
            
        except KeyboardInterrupt:
            print("\n⚠️ Optimization interrupted by user")
    
    def _save_results(self):
        """Save results to JSON file"""
        output_file = Path('hyperparameter_results_optuna.json')
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def save_visualizations(self):
        """Generate and save Optuna visualizations"""
        try:
            import plotly
            
            viz_dir = Path('optuna_visualizations')
            viz_dir.mkdir(exist_ok=True)
            
            print("\n3. Generating visualizations...")
            
            # Optimization history
            fig1 = optuna.visualization.plot_optimization_history(self.study)
            fig1.write_html(str(viz_dir / 'optimization_history.html'))
            print("   ✓ Optimization history saved")
            
            # Parameter importances
            try:
                fig2 = optuna.visualization.plot_param_importances(self.study)
                fig2.write_html(str(viz_dir / 'param_importances.html'))
                print("   ✓ Parameter importances saved")
            except:
                print("   ⚠️ Not enough trials for importance plot")
            
            # Slice plot
            fig3 = optuna.visualization.plot_slice(self.study)
            fig3.write_html(str(viz_dir / 'param_slice.html'))