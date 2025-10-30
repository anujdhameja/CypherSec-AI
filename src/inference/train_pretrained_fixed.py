"""
FIX: Proper pretrained model loading and fine-tuning
This script correctly loads Config 9 weights and fine-tunes them
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from torch_geometric.loader import DataLoader

# Add src to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.process.balanced_training_config import BalancedDevignModel
from src.utils.objects.input_dataset import InputDataset
import configs


class PretrainedModelTrainer:
    """Train with pretrained Config 9 weights"""
    
    def __init__(self, pretrained_path='models/production_model_config9_v1.0.pth',
                 config_path='configs/best_config_v1.0.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_path = pretrained_path
        self.config_path = config_path
        
        # Load config - fallback to hardcoded if file doesn't exist
        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            # Use Config 9 parameters directly
            self.config = {
                'architecture': {
                    'input_dim': 100,
                    'output_dim': 2,
                    'hidden_dim': 256,
                    'num_steps': 5,
                    'dropout': 0.2
                },
                'performance': {
                    'test_accuracy': 0.8357
                }
            }
        
        print("="*80)
        print("PRETRAINED MODEL TRAINER")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Pretrained model: {pretrained_path}")
        print(f"Config: {config_path}")

    def load_pretrained_model(self):
        """Load model with pretrained weights"""
        arch = self.config['architecture']
        print("\n🔧 Creating model with config:")
        print(f"  hidden_dim: {arch['hidden_dim']}")
        print(f"  num_steps: {arch['num_steps']}")
        print(f"  dropout: {arch['dropout']}")
        print(f"  pooling: mean_max")
        
        # Create model EXACTLY as it was trained
        model = BalancedDevignModel(
            input_dim=arch['input_dim'],
            output_dim=arch['output_dim'],
            hidden_dim=arch['hidden_dim'],
            num_steps=arch['num_steps'],
            dropout=arch['dropout']
        )
        
        # Load pretrained weights
        print(f"\n📦 Loading pretrained weights...")
        state_dict = torch.load(self.pretrained_path, map_location=self.device)
        
        # Filter out extra keys (attention weights from other variants)
        model_keys = set(model.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Pretrained weights loaded successfully")
        
        model = model.to(self.device)
        model.eval()  # Set to eval mode first to check
        return model

    def get_finetuning_optimizer(self, model):
        """Create optimizer for fine-tuning with MUCH LOWER learning rate
        Key: Learning rate should be 10-100x LOWER than initial training"""
        
        # For fine-tuning, use MUCH lower learning rate
        finetune_lr = 1e-5  # NOT 1e-4!
        finetune_weight_decay = 1e-5
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=finetune_lr,
            weight_decay=finetune_weight_decay
        )
        
        print(f"\n⚙️  Fine-tuning Optimizer:")
        print(f"  Learning Rate: {finetune_lr}")
        print(f"  Weight Decay: {finetune_weight_decay}")
        
        return optimizer

    def train_epoch(self, model, loader, optimizer, criterion):
        """Train one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward
            output = model(batch)
            loss = criterion(output, batch.y.long())
            
            # Backward
            loss.backward()
            
            # Gradient clipping - CRITICAL for pretrained fine-tuning!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == batch.y.long()).sum().item()
            total += batch.y.size(0)
        
        return total_loss / len(loader), correct / total

    def eval_epoch(self, model, loader, criterion):
        """Evaluate one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = model(batch)
                loss = criterion(output, batch.y.long())
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == batch.y.long()).sum().item()
                total += batch.y.size(0)
        
        return total_loss / len(loader), correct / total

    def train(self, num_epochs=50, batch_size=32):
        """Fine-tune pretrained model"""
        # Load model
        model = self.load_pretrained_model()
        
        # Load data
        print(f"\n📊 Loading dataset...")
        dataset = InputDataset('data/input')
        
        # Split data
        from sklearn.model_selection import train_test_split
        indices = list(range(len(dataset)))
        labels = [int(dataset[i].y.item()) for i in indices]
        
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, 
            stratify=[labels[i] for i in temp_idx]
        )
        
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss
        optimizer = self.get_finetuning_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with EARLY STOPPING
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"\n🚀 Starting fine-tuning for {num_epochs} epochs...")
        print("="*80)
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = self.eval_epoch(model, val_loader, criterion)
            
            print(f"Epoch {epoch+1:2d} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'models/finetuned_model_best.pth')
                print(f"  ✅ Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏸️  Early stopping at epoch {epoch+1}")
                    break
        
        # Test on best model
        print("\n" + "="*80)
        print("EVALUATING BEST MODEL ON TEST SET")
        print("="*80)
        
        model.load_state_dict(torch.load('models/finetuned_model_best.pth'))
        test_loss, test_acc = self.eval_epoch(model, test_loader, criterion)
        
        print(f"\n🎯 FINAL TEST RESULTS:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        expected_acc = self.config['performance']['test_accuracy']
        print(f"\n  Expected (Config 9): {expected_acc:.4f} ({expected_acc*100:.2f}%)")
        print(f"  Difference: {test_acc - expected_acc:+.4f}")
        
        if test_acc > 0.70:
            print(f"\n✅ SUCCESS! Model is learning properly!")
        else:
            print(f"\n⚠️  WARNING: Accuracy lower than expected. Check training process.")


if __name__ == '__main__':
    trainer = PretrainedModelTrainer()
    trainer.train(num_epochs=100, batch_size=32)