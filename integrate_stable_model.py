"""
Integration Script: How to use StableDevignModel with your existing code
"""

import torch
import torch.nn as nn
from stable_training import StableDevignModel, train_one_epoch_stable, validate_stable, stable_training_config


def create_stable_devign_step(path: str, device: str, model_config: dict, learning_rate: float, weight_decay: float, loss_lambda: float):
    """
    Create a stable version of your Devign step class
    This replaces the existing Devign class in src/process/devign.py
    """
    
    class StableDevignStep:
        def __init__(self):
            self.path = path
            self.lr = learning_rate
            self.wd = weight_decay
            self.ll = loss_lambda
            
            print(f"🔧 STABLE Devign Setup:")
            print(f"   LR: {self.lr}; WD: {self.wd}; LL: {self.ll}")
            
            # Create stable model
            self.model = StableDevignModel(
                input_dim=205,
                hidden_dim=200,
                num_steps=4,  # REDUCED from 8
                dropout=0.3
            ).to(device)
            
            # Simple CrossEntropy loss (no complex lambda loss for stability)
            self.criterion = nn.CrossEntropyLoss()
            
            # Optimizer
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.lr, 
                weight_decay=self.wd
            )
            
            # Learning rate scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            
            self.device = device
            self.count_parameters()
        
        def __call__(self, i, batch, target):
            """Training step - replaces your existing step function"""
            # Move to device
            batch = batch.to(self.device)
            target = target.to(self.device)
            
            # Use stable training function
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward
            output = self.model(batch)
            target_clean = target.squeeze().long()
            
            # Loss
            loss = self.criterion(output, target_clean)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️ NaN loss at step {i}, skipping")
                return None
            
            # Backward with gradient clipping
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Check for exploding gradients
            if grad_norm > 10.0:
                print(f"⚠️ Large gradient norm: {grad_norm:.2f} at step {i}")
            
            self.optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct = (pred == target_clean).sum().item()
            total = target_clean.size(0)
            accuracy = correct / total if total > 0 else 0
            
            # Return stats (matching your existing interface)
            class Stats:
                def __init__(self, loss, accuracy, total):
                    self.loss = loss
                    self.accuracy = accuracy
                    self.total = total
            
            return Stats(loss.item(), accuracy, total)
        
        def validate_step(self, batch, target):
            """Validation step"""
            batch = batch.to(self.device)
            target = target.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(batch)
                target_clean = target.squeeze().long()
                loss = self.criterion(output, target_clean)
                
                pred = output.argmax(dim=1)
                correct = (pred == target_clean).sum().item()
                total = target_clean.size(0)
                accuracy = correct / total if total > 0 else 0
                
                class Stats:
                    def __init__(self, loss, accuracy, total):
                        self.loss = loss
                        self.accuracy = accuracy
                        self.total = total
                
                return Stats(loss.item(), accuracy, total)
        
        def step_scheduler(self, val_loss):
            """Step the learning rate scheduler"""
            self.scheduler.step(val_loss)
        
        def load(self):
            """Load model"""
            self.model.load(self.path)
        
        def save(self):
            """Save model"""
            self.model.save(self.path)
        
        def count_parameters(self):
            count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"✓ The stable model has {count:,} trainable parameters")
    
    return StableDevignStep()


def modify_your_devign_py():
    """
    Instructions for modifying src/process/devign.py
    """
    print("\n" + "="*80)
    print("HOW TO INTEGRATE STABLE MODEL")
    print("="*80)
    
    print("\n📝 Option 1: Replace your Devign class")
    print("   1. Backup your current src/process/devign.py")
    print("   2. Replace the Devign class with StableDevignStep from above")
    print("   3. Update imports to include stable_training")
    
    print("\n📝 Option 2: Create new stable config")
    print("   1. Copy your config.json to config_stable.json")
    print("   2. Update model parameters:")
    print("      - Set num_layers: 4 (instead of 8)")
    print("      - Set learning_rate: 3e-4")
    print("      - Set epochs: 50")
    
    print("\n📝 Option 3: Quick test (recommended)")
    print("   1. Run: python stable_training.py")
    print("   2. If it works, integrate step by step")
    
    print("\n🔧 Key Changes Needed:")
    print("   - Replace GatedGraphConv num_layers: 8 → 4")
    print("   - Add gradient clipping (max_norm=1.0)")
    print("   - Use simple CrossEntropyLoss")
    print("   - Add learning rate scheduler")
    print("   - Add batch normalization")


def create_stable_config_json():
    """Create a stable version of config.json"""
    stable_config = {
        "model": {
            "gated_graph_conv_args": {
                "out_channels": 200,
                "num_layers": 4,  # REDUCED from 8
                "aggr": "add",
                "bias": True
            },
            "conv_args": {
                "conv1d_1": {
                    "in_channels": 205,
                    "out_channels": 50,
                    "kernel_size": 3
                },
                "conv1d_2": {
                    "in_channels": 50,
                    "out_channels": 20,
                    "kernel_size": 1
                }
            }
        },
        "train": {
            "learning_rate": 3e-4,  # Increased from 1e-4
            "weight_decay": 1.3e-6,
            "loss_lambda": 0.0,  # Disabled for stability
            "epochs": 50,  # Reduced from 100
            "batch_size": 8,
            "shuffle": False,
            "gradient_clip": 1.0  # NEW
        }
    }
    
    import json
    with open('config_stable.json', 'w') as f:
        json.dump(stable_config, f, indent=2)
    
    print("\n✓ Created config_stable.json with stable parameters")
    return stable_config


if __name__ == "__main__":
    print("🚀 STABLE MODEL INTEGRATION GUIDE")
    
    # Show config
    config = stable_training_config()
    
    # Show integration steps
    modify_your_devign_py()
    
    # Create stable config
    stable_config = create_stable_config_json()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Test the stable model:")
    print("   python stable_training.py")
    print("\n2. If it works, try with your data:")
    print("   - Modify your main.py to use StableDevignModel")
    print("   - Or use config_stable.json")
    print("\n3. Expected results:")
    print("   - Smoother loss curves")
    print("   - No gradient explosions")
    print("   - Accuracy should reach 50-55% by epoch 20")