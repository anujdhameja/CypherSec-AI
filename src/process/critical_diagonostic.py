"""
CRITICAL DIAGNOSTIC
Find out WHY validation accuracy is stuck at exactly 49.73%
"""

import torch
import torch.nn as nn
import pandas as pd
from collections import Counter
from pathlib import Path


def diagnose_model_outputs(model, val_loader, device):
    """
    Check what the model is actually outputting
    """
    print("\n" + "="*80)
    print("MODEL OUTPUT DIAGNOSIS")
    print("="*80)
    
    model.eval()
    
    all_outputs = []
    all_predictions = []
    all_targets = []
    all_softmax = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = batch.to(device)
            output = model(batch)
            target = batch.y.squeeze().long()
            
            # Raw logits
            all_outputs.append(output.cpu())
            
            # Predictions
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().tolist())
            
            # Targets
            all_targets.extend(target.cpu().tolist())
            
            # Softmax probabilities
            softmax = torch.softmax(output, dim=1)
            all_softmax.append(softmax.cpu())
            
            # Show first batch in detail
            if i == 0:
                print(f"\n📊 First Batch Details:")
                print(f"   Batch size: {len(target)}")
                print(f"\n   Raw logits (first 5):")
                for j in range(min(5, len(output))):
                    print(f"      Sample {j}: [{output[j, 0]:.4f}, {output[j, 1]:.4f}]")
                
                print(f"\n   After softmax (first 5):")
                for j in range(min(5, len(softmax))):
                    print(f"      Sample {j}: [Class0: {softmax[j, 0]:.4f}, Class1: {softmax[j, 1]:.4f}]")
                
                print(f"\n   Predictions vs Targets (first 10):")
                for j in range(min(10, len(pred))):
                    symbol = "✓" if pred[j] == target[j] else "✗"
                    print(f"      {symbol} Pred: {pred[j]}, Target: {target[j]}")
    
    # Aggregate statistics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_softmax = torch.cat(all_softmax, dim=0)
    
    print(f"\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    
    # Prediction distribution
    pred_counts = Counter(all_predictions)
    target_counts = Counter(all_targets)
    
    print(f"\n📊 Prediction Distribution:")
    print(f"   Class 0: {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(all_predictions):.2%})")
    print(f"   Class 1: {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(all_predictions):.2%})")
    print(f"   Unique predictions: {set(all_predictions)}")
    
    print(f"\n📊 Target Distribution:")
    print(f"   Class 0: {target_counts[0]} ({target_counts[0]/len(all_targets):.2%})")
    print(f"   Class 1: {target_counts[1]} ({target_counts[1]/len(all_targets):.2%})")
    
    # Output statistics
    print(f"\n📊 Raw Output Statistics:")
    print(f"   Class 0 logits: mean={all_outputs[:, 0].mean():.4f}, std={all_outputs[:, 0].std():.4f}")
    print(f"   Class 1 logits: mean={all_outputs[:, 1].mean():.4f}, std={all_outputs[:, 1].std():.4f}")
    print(f"   Difference (0-1): mean={( all_outputs[:, 0] - all_outputs[:, 1]).mean():.4f}")
    
    print(f"\n📊 Softmax Probability Statistics:")
    print(f"   Class 0 prob: mean={all_softmax[:, 0].mean():.4f}, std={all_softmax[:, 0].std():.4f}")
    print(f"   Class 1 prob: mean={all_softmax[:, 1].mean():.4f}, std={all_softmax[:, 1].std():.4f}")
    
    # Diagnosis
    print(f"\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    unique_preds = len(set(all_predictions))
    
    if unique_preds == 1:
        predicted_class = list(set(all_predictions))[0]
        print(f"\n❌ PROBLEM IDENTIFIED:")
        print(f"   Model predicts ONLY class {predicted_class} for ALL samples!")
        print(f"\n   This is why accuracy is stuck at {target_counts[predicted_class]/len(all_targets):.2%}")
        print(f"   (That's the proportion of class {predicted_class} in validation set)")
        
        # Why is this happening?
        print(f"\n💡 Likely Causes:")
        logit_diff = (all_outputs[:, 0] - all_outputs[:, 1]).mean()
        if abs(logit_diff) > 2:
            print(f"   1. ✓ Output bias detected (logit difference: {logit_diff:.4f})")
            print(f"      Model consistently outputs higher logits for class {predicted_class}")
        
        if all_softmax[:, predicted_class].std() < 0.1:
            print(f"   2. ✓ Low output variance (std: {all_softmax[:, predicted_class].std():.4f})")
            print(f"      Model outputs are too similar - not discriminating between samples")
        
        print(f"\n🔧 Suggested Fixes:")
        print(f"   1. Check weight initialization")
        print(f"   2. Reduce regularization (dropout/weight_decay)")
        print(f"   3. Increase learning rate")
        print(f"   4. Check if model layers are frozen")
        print(f"   5. Verify loss function is working")
        
    elif unique_preds == 2:
        print(f"\n✓ Model IS predicting both classes")
        print(f"   But validation accuracy is still low...")
        
        # Calculate confusion matrix
        correct = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
        acc = correct / len(all_targets)
        print(f"\n   Actual accuracy: {acc:.2%}")
        
        if acc < 0.55:
            print(f"\n⚠️ Model is barely better than random")
            print(f"💡 Possible causes:")
            print(f"   1. Model too simple (underfitting)")
            print(f"   2. Learning rate too low")
            print(f"   3. Not enough training epochs")
    
    return {
        'unique_predictions': unique_preds,
        'pred_distribution': dict(pred_counts),
        'target_distribution': dict(target_counts),
        'output_stats': {
            'class0_mean': all_outputs[:, 0].mean().item(),
            'class1_mean': all_outputs[:, 1].mean().item(),
        }
    }


def check_model_gradients(model, train_loader, device):
    """
    Check if gradients are flowing properly
    """
    print("\n" + "="*80)
    print("GRADIENT FLOW CHECK")
    print("="*80)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Get one batch
    batch = next(iter(train_loader)).to(device)
    
    # Forward pass
    output = model(batch)
    target = batch.y.squeeze().long()
    loss = criterion(output, target)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check gradients
    print(f"\n📊 Gradient Statistics:")
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            print(f"\n   {name}:")
            print(f"      Norm: {grad_norm:.6f}")
            print(f"      Mean: {grad_mean:.6f}")
            print(f"      Std:  {grad_std:.6f}")
            
            if grad_norm < 1e-6:
                print(f"      ⚠️ WARNING: Very small gradients (vanishing?)")
            elif grad_norm > 100:
                print(f"      ⚠️ WARNING: Very large gradients (exploding?)")
        else:
            print(f"\n   {name}: ❌ NO GRADIENT (layer might be frozen!)")


def check_data_distribution(train_loader, val_loader):
    """
    Check if train and val have similar distributions
    """
    print("\n" + "="*80)
    print("DATA DISTRIBUTION CHECK")
    print("="*80)
    
    # Collect targets
    train_targets = []
    val_targets = []
    
    for batch in train_loader:
        train_targets.extend(batch.y.squeeze().long().tolist())
    
    for batch in val_loader:
        val_targets.extend(batch.y.squeeze().long().tolist())
    
    train_counts = Counter(train_targets)
    val_counts = Counter(val_targets)
    
    print(f"\n📊 Training Set:")
    print(f"   Total: {len(train_targets)}")
    print(f"   Class 0: {train_counts[0]} ({train_counts[0]/len(train_targets):.2%})")
    print(f"   Class 1: {train_counts[1]} ({train_counts[1]/len(train_targets):.2%})")
    print(f"   Balance: {train_counts[0]/train_counts[1]:.2f}")
    
    print(f"\n📊 Validation Set:")
    print(f"   Total: {len(val_targets)}")
    print(f"   Class 0: {val_counts[0]} ({val_counts[0]/len(val_targets):.2%})")
    print(f"   Class 1: {val_counts[1]} ({val_counts[1]/len(val_targets):.2%})")
    print(f"   Balance: {val_counts[0]/val_counts[1]:.2f}")
    
    # Check for imbalance
    if abs(train_counts[0]/train_counts[1] - 1.0) > 0.3:
        print(f"\n⚠️ Training set is imbalanced!")
        print(f"   Consider using class weights in loss function")
    
    if abs(val_counts[0]/val_counts[1] - 1.0) > 0.3:
        print(f"\n⚠️ Validation set is imbalanced!")


def run_full_diagnostic(model, train_loader, val_loader, device):
    """
    Run complete diagnostic suite
    """
    print("\n" + "#"*80)
    print("# COMPLETE DIAGNOSTIC SUITE")
    print("#"*80)
    
    # 1. Check data
    check_data_distribution(train_loader, val_loader)
    
    # 2. Check gradients
    check_model_gradients(model, train_loader, device)
    
    # 3. Check outputs
    results = diagnose_model_outputs(model, val_loader, device)
    
    # Final recommendation
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)
    
    if results['unique_predictions'] == 1:
        print("\n🎯 PRIMARY ISSUE: Model predicting only one class")
        print("\n📋 Action Plan:")
        print("   1. FIRST: Reduce dropout to 0.2")
        print("   2. SECOND: Reduce weight_decay to 1e-6")
        print("   3. THIRD: Increase learning_rate to 5e-4")
        print("   4. VERIFY: Check if model trains on a SINGLE batch:")
        print("""
   # Single batch overfitting test:
   batch = next(iter(train_loader))
   for i in range(100):
       loss = train_one_step(model, batch)
       if i % 10 == 0:
           print(f"Step {i}: Loss {loss:.4f}")
   # If loss doesn't decrease → model architecture issue
   # If loss decreases → regularization too strong
""")
    
    return results


# ============================================
# SINGLE BATCH OVERFITTING TEST
# ============================================

def single_batch_overfit_test(model, train_loader, device, num_steps=100):
    """
    Critical test: Can model overfit a SINGLE batch?
    
    If YES → Model architecture is fine, regularization is the issue
    If NO  → Model architecture has a problem
    """
    print("\n" + "="*80)
    print("SINGLE BATCH OVERFITTING TEST")
    print("="*80)
    print("\nThis test checks if model CAN learn at all")
    print("by trying to overfit one batch completely.")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # High LR
    criterion = nn.CrossEntropyLoss()
    
    # Get one batch
    batch = next(iter(train_loader)).to(device)
    target = batch.y.squeeze().long()
    
    print(f"\nBatch size: {batch.num_graphs}")
    print(f"Target distribution: {Counter(target.cpu().tolist())}")
    
    print(f"\nTraining on single batch for {num_steps} steps...")
    print("(Model should reach 100% accuracy if working properly)\n")
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        output = model(batch)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean().item()
        
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step:3d}: Loss={loss.item():.4f}, Acc={acc:.2%}, "
                  f"Unique preds={len(pred.unique())}")
    
    # Final check
    print(f"\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    final_acc = acc
    final_unique_preds = len(pred.unique())
    
    if final_acc > 0.95 and final_unique_preds == 2:
        print(f"\n✓ PASS: Model CAN learn!")
        print(f"   Final accuracy: {final_acc:.2%}")
        print(f"   Predicting both classes: Yes")
        print(f"\n💡 Diagnosis: Model architecture is FINE")
        print(f"   Problem is likely: Too much regularization in full training")
        print(f"\n🔧 Solution:")
        print(f"   - Reduce dropout (try 0.2)")
        print(f"   - Reduce weight_decay (try 1e-6)")
        print(f"   - Keep learning_rate same or slightly higher")
        
    elif final_unique_preds == 1:
        print(f"\n❌ FAIL: Model still predicting only ONE class")
        print(f"   Even with 100 steps on same batch!")
        print(f"\n💡 Diagnosis: Model architecture issue OR initialization issue")
        print(f"\n🔧 Possible solutions:")
        print(f"   1. Check model initialization")
        print(f"   2. Check if any layers are frozen")
        print(f"   3. Try simpler model (fewer GNN steps)")
        print(f"   4. Check loss function is correct")
        
    else:
        print(f"\n⚠️ PARTIAL: Model is learning but slowly")
        print(f"   Final accuracy: {final_acc:.2%}")
        print(f"   Predicting both classes: {final_unique_preds == 2}")
        print(f"\n💡 Diagnosis: Model CAN learn but something is limiting it")
        print(f"\n🔧 Try:")
        print(f"   - Increase learning rate")
        print(f"   - Reduce GNN steps (simpler model)")


# ============================================
# MAIN INTEGRATION SCRIPT
# ============================================

def create_diagnostic_script():
    """
    Create standalone diagnostic script for easy use
    """
    script = '''
"""
Run This Diagnostic Script
Integrates with your existing training code
"""

import os
import sys
import torch
from torch_geometric.loader import DataLoader
import pandas as pd

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(script_dir))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import your model
try:
    from src.process.model import DevignModel  # Adjust import as needed
    from src.process.balanced_training_config import BalancedDevignModel
    # Diagnostic functions are already imported at the top
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Current Python path:", sys.path)
    raise

def main():
    print("\\n" + "#"*80)
    print("# DIAGNOSTIC RUN")
    print("#"*80)
    
    # Load data
    print("\\nLoading data...")
    
    # Load your processed input files
    import sys
    sys.path.append('.')
    from src import data
    from paths import PATHS
    
    input_dataset = data.loads(PATHS.input)
    
    # Split data (same as your training)
    train_dataset, val_dataset, test_dataset = data.train_val_test_split(
        input_dataset, shuffle=False
    )
    
    # Create loaders
    train_loader = train_dataset.get_loader(batch_size=8, shuffle=False)
    val_loader = val_dataset.get_loader(batch_size=8, shuffle=False)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create model
    print("\\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BalancedDevignModel(
        input_dim=205,
        output_dim=2,
        hidden_dim=200,
        num_steps=4,
        dropout=0.4
    ).to(device)
    
    # Try to load trained weights if they exist
    try:
        model.load('data/model/devign.model')
        print("✓ Loaded trained model")
    except:
        print("⚠️ Using random initialization (no trained model found)")
    
    # Run diagnostics
    print("\\n" + "="*80)
    print("RUNNING DIAGNOSTICS")
    print("="*80)
    
    # Full diagnostic
    results = run_full_diagnostic(model, train_loader, val_loader, device)
    
    # Single batch test
    print("\\n" + "="*80)
    single_batch_overfit_test(model, train_loader, device, num_steps=100)
    
    print("\\n" + "#"*80)
    print("# DIAGNOSTIC COMPLETE")
    print("#"*80)
    print("\\nCheck the output above for issues and recommendations.")


if __name__ == "__main__":
    main()
'''
    
    with open('run_diagnostics.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("\n✓ Created: run_diagnostics.py")
    print("Run: python run_diagnostics.py")


if __name__ == "__main__":
    print(__doc__)
    create_diagnostic_script()