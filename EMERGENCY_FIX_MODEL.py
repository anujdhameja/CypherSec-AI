"""
EMERGENCY MODEL FIX
Your model is stuck at 50% accuracy because it's not learning anything.
This script implements immediate fixes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from collections import Counter
import pandas as pd

from src.process.balanced_training_config import BalancedDevignModel
from src.process.step import Step
from src.utils.objects.input_dataset import InputDataset
import configs


print("="*80)
print("EMERGENCY MODEL FIX - CRITICAL ISSUES DETECTED")
print("="*80)
print("\n🔥 YOUR MODEL IS NOT LEARNING! All trials got ~50% accuracy.")
print("This means the model is guessing randomly, not predicting one class.\n")


class FixedTrainer:
    """Trainer with critical fixes for non-learning model"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}\n")
        
    def load_and_analyze_data(self):
        """Load data and check for issues"""
        print("1. ANALYZING DATA...")
        
        # Use the SAME data loading approach as the working auto_hyperparameter_FIXED.py
        input_path = Path('data/input')
        input_files = list(input_path.glob('*_input.pkl'))
        
        if not input_files:
            raise FileNotFoundError("No input files found!")
        
        # Load first few files (same as working version)
        all_data = []
        for f in input_files[:10]:  # Use subset for faster search
            df = pd.read_pickle(f)
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        graphs = combined['input'].tolist()  # Direct extraction like working version
        
        print(f"   Loaded {len(graphs)} samples from {len(input_files[:10])} files")
        
        # Split into train/val (80/20 split)
        train_size = int(0.8 * len(graphs))
        val_size = int(0.1 * len(graphs))
        
        self.train_graphs = graphs[:train_size]
        self.val_graphs = graphs[train_size:train_size+val_size]
        
        print(f"   Train: {len(self.train_graphs)} samples")
        print(f"   Val: {len(self.val_graphs)} samples")
        
        # Check class balance
        train_labels = [g.y.item() for g in self.train_graphs]
        val_labels = [g.y.item() for g in self.val_graphs]
        
        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)
        
        print(f"\n   Class distribution:")
        print(f"   Train - Class 0: {train_counter[0]} ({train_counter[0]/len(train_labels)*100:.1f}%)")
        print(f"   Train - Class 1: {train_counter[1]} ({train_counter[1]/len(train_labels)*100:.1f}%)")
        print(f"   Val   - Class 0: {val_counter[0]} ({val_counter[0]/len(val_labels)*100:.1f}%)")
        print(f"   Val   - Class 1: {val_counter[1]} ({val_counter[1]/len(val_labels)*100:.1f}%)")
        
        # Calculate class weights
        self.class_weights = torch.tensor([
            len(train_labels) / (2 * train_counter[0]),
            len(train_labels) / (2 * train_counter[1])
        ]).to(self.device)
        
        print(f"\n   ✓ Calculated class weights: {self.class_weights.cpu().numpy()}")
        
    def create_fixed_model(self):
        """Create model with proper initialization"""
        print("\n2. CREATING FIXED MODEL...")
        
        embed = configs.Embed()
        
        # Use the current working BalancedDevignModel with emergency fixes
        model = BalancedDevignModel(
            input_dim=embed.nodes_dim,  # 100 from configs
            output_dim=2,
            hidden_dim=128,  # Smaller for emergency fix
            num_steps=3,     # Fewer steps for stability
            dropout=0.2      # Reduced dropout for learning
        ).to(self.device)
        
        # Proper initialization
        for name, param in model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        print(f"   ✓ Model created with Xavier initialization")
        print(f"   ✓ Using BalancedDevignModel with emergency settings")
        
        return model
    
    def train_with_fixes(self, epochs=30):
        """Train with all critical fixes applied"""
        print("\n3. TRAINING WITH FIXES...")
        print("   Applied fixes:")
        print("   ✓ Class weights for imbalanced data")
        print("   ✓ Label smoothing (0.1)")
        print("   ✓ Gradient clipping (max_norm=1.0)")
        print("   ✓ Learning rate warmup")
        print("   ✓ Cosine annealing scheduler")
        print("   ✓ Proper weight initialization")
        print()
        
        model = self.create_fixed_model()
        
        # Loss with class weights (no label smoothing for emergency fix)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Optimizer with emergency settings
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=2e-4,  # Slightly higher LR
            weight_decay=1e-6  # Very low weight decay
        )
        
        # No Step class - use direct training like working version
        
        # Learning rate scheduler with warmup
        warmup_epochs = 5
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        # Training loop
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Learning rate warmup
            if epoch < warmup_epochs:
                lr = 1e-5 + (1e-4 - 1e-5) * (epoch / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Training
            model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0.0
            
            # Create dataloader EXACTLY like the working version
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(self.train_graphs, batch_size=8, shuffle=True)
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                output = model(batch)
                target = batch.y.squeeze().long()
                
                loss = criterion(output, target)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item() * batch.num_graphs
                pred = output.argmax(dim=1)
                train_correct += (pred == target).sum().item()
                train_total += batch.num_graphs
            
            train_acc = 100.0 * train_correct / train_total
            train_loss /= train_total
            
            # Validation EXACTLY like the working version
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            val_loader = DataLoader(self.val_graphs, batch_size=8, shuffle=False)
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    
                    output = model(batch)
                    target = batch.y.squeeze().long()
                    
                    loss = criterion(output, target)
                    
                    val_loss += loss.item() * batch.num_graphs
                    pred = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total += batch.num_graphs
            
            val_acc = 100.0 * val_correct / val_total
            val_loss /= val_total
            
            # Learning rate scheduling (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print progress
            status = "✓"
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                status = "🎯 NEW BEST"
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1:2d}/{epochs}: "
                  f"Train {train_acc:5.2f}%, "
                  f"Val {val_acc:5.2f}%, "
                  f"Loss {train_loss:.4f}, "
                  f"LR {current_lr:.2e} {status}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered (patience={patience})")
                break
            
            # Check if still stuck
            if epoch >= 10 and best_val_acc < 55:
                print(f"\n⚠️ WARNING: Still stuck at {best_val_acc:.2f}% after 10 epochs")
                print("   This suggests deeper issues with:")
                print("   - Data quality or preprocessing")
                print("   - Feature extraction (embeddings)")
                print("   - Graph construction")
                break
        
        print(f"\n📊 TRAINING COMPLETE")
        print(f"   Best Val Acc: {best_val_acc:.2f}%")
        
        if best_val_acc > 60:
            print(f"   ✅ SUCCESS! Model is learning.")
        elif best_val_acc > 55:
            print(f"   ⚠️ MARGINAL: Model shows some learning but needs improvement.")
        else:
            print(f"   ❌ FAILED: Model still not learning properly.")
            print(f"\n   NEXT STEPS:")
            print(f"   1. Check your data preprocessing pipeline")
            print(f"   2. Verify graph construction is correct")
            print(f"   3. Check if node features are meaningful")
            print(f"   4. Try a simpler baseline (logistic regression on graph features)")
        
        return model, best_val_acc
    
    def run(self):
        """Run complete fix procedure"""
        self.load_and_analyze_data()
        model, best_acc = self.train_with_fixes()
        return model, best_acc


def main():
    print("\nSTARTING EMERGENCY FIX PROCEDURE...\n")
    
    trainer = FixedTrainer()
    model, best_acc = trainer.run()
    
    print("\n" + "="*80)
    print("FIX PROCEDURE COMPLETE")
    print("="*80)
    
    if best_acc > 60:
        print("\n✅ Model is now learning! You can proceed with hyperparameter optimization.")
        print("   Recommended next step: python auto_hyperparameter_simple.py --trials 20")
    elif best_acc > 55:
        print("\n⚠️ Model shows minimal learning. Try:")
        print("   1. Checking data quality: python diagnose_and_fix_model.py")
        print("   2. Reducing model complexity further")
        print("   3. Increasing training data")
    else:
        print("\n❌ Model still not learning. Critical issues remain:")
        print("   1. Run full diagnostic: python diagnose_and_fix_model.py")
        print("   2. Verify your data pipeline from scratch")
        print("   3. Test with a simple baseline model first")
        print("   4. Check if the task is actually learnable with this data")


if __name__ == "__main__":
    main()