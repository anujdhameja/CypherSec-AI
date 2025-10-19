"""
Configuration Validator
Checks if your config matches the README's working setup
"""

import sys
import os


def check_config_params():
    """Check if config matches README recommendations"""
    print("="*80)
    print("CONFIGURATION VALIDATION")
    print("="*80)
    
    try:
        sys.path.insert(0, os.getcwd())
        import configs
        
        # Get configs
        devign = configs.Devign()
        process = configs.Process()
        
        print("\n📋 Current Configuration:")
        print(f"   Learning Rate: {devign.learning_rate}")
        print(f"   Weight Decay: {devign.weight_decay}")
        print(f"   Loss Lambda: {devign.loss_lambda}")
        print(f"   Epochs: {process.epochs}")
        print(f"   Batch Size: {process.batch_size}")
        print(f"   Shuffle: {process.shuffle}")
        
        print("\n📚 README Recommended:")
        print(f"   Learning Rate: 1e-4 (0.0001)")
        print(f"   Weight Decay: 1.3e-6")
        print(f"   Loss Lambda: 1.3e-6")
        print(f"   Epochs: 30-100")
        print(f"   Batch Size: 8")
        print(f"   Shuffle: False")
        
        # Check for issues
        issues = []
        warnings = []
        
        # Learning rate check
        if devign.learning_rate > 1e-3:
            warnings.append("⚠️ Learning rate might be too high (README uses 1e-4)")
        elif devign.learning_rate < 1e-5:
            warnings.append("⚠️ Learning rate might be too low (README uses 1e-4)")
        elif abs(devign.learning_rate - 1e-4) < 1e-5:
            print("\n   ✓ Learning rate matches README")
        
        # Epoch check
        if process.epochs < 30:
            warnings.append(f"⚠️ Epochs ({process.epochs}) lower than README (30-100)")
        
        # Batch size check
        if process.batch_size != 8:
            warnings.append(f"⚠️ Batch size ({process.batch_size}) differs from README (8)")
        
        # Print issues
        if warnings:
            print("\n⚠️ Configuration Warnings:")
            for w in warnings:
                print(f"   {w}")
        
        return devign, process
        
    except Exception as e:
        print(f"❌ Error loading configs: {e}")
        return None, None


def check_loss_calculation():
    """Check how loss is calculated"""
    print("\n" + "="*80)
    print("LOSS FUNCTION CHECK")
    print("="*80)
    
    print("\n🔍 The README mentions 'loss_lambda' parameter.")
    print("This suggests a composite loss function like:")
    print("\n   total_loss = classification_loss + lambda * regularization_loss")
    print("\n📝 Make sure your loss is calculated as:")
    print("   ```python")
    print("   ce_loss = criterion(output, target)")
    print("   reg_loss = # some regularization (e.g., L2 on weights)")
    print("   total_loss = ce_loss + loss_lambda * reg_loss")
    print("   ```")


def analyze_readme_results():
    """Analyze what the README results tell us"""
    print("\n" + "="*80)
    print("README PERFORMANCE ANALYSIS")
    print("="*80)
    
    print("\n📊 With Early Stopping (5 epochs):")
    print("   TP: 37, FP: 27, TN: 22, FN: 15")
    print("   Accuracy: 58.4%")
    print("   Recall: 71.2% (finds most vulnerabilities)")
    print("   Precision: 57.8% (many false alarms)")
    
    print("\n📊 Without Early Stopping (30 epochs):")
    print("   TP: 38, FP: 34, TN: 15, FN: 14")
    print("   Accuracy: 52.5%")
    print("   Recall: 73.1%")
    print("   Precision: 52.8%")
    
    print("\n💡 Key Insights:")
    print("   1. This is a HARD problem - best accuracy is only ~58%")
    print("   2. Model is biased toward predicting 'vulnerable'")
    print("   3. Early stopping helps (5 epochs better than 30!)")
    print("   4. Sample dataset is very small (101 test samples)")
    
    print("\n🎯 Expected Learning Curve:")
    print("   Epoch 1-5:   Acc might be 30-40%")
    print("   Epoch 5-10:  Acc should reach 50-55%")
    print("   Epoch 10-30: Acc might plateau at 55-60%")


def diagnose_current_performance():
    """Diagnose why current performance is low"""
    print("\n" + "="*80)
    print("CURRENT PERFORMANCE DIAGNOSIS")
    print("="*80)
    
    print("\n🔴 Your Results After 2 Epochs:")
    print("   Train Acc: 8-27%")
    print("   Val Acc: 27%")
    
    print("\n🤔 This is VERY low, even for early training.")
    print("\n⚠️ Most Likely Issues:")
    
    issues = [
        {
            'issue': 'Target Shape/Type Mismatch',
            'probability': '90%',
            'check': '''
# In your training loop, add this:
print(f"Target shape: {target.shape}")  # Should be [batch_size]
print(f"Target dtype: {target.dtype}")  # Should be torch.long
print(f"Target values: {target[:5]}")   # Should be 0s and 1s
print(f"Output shape: {output.shape}")  # Should be [batch_size, 2]
''',
            'fix': '''
# Before loss calculation:
target = target.squeeze().long()  # Ensure correct shape and type
'''
        },
        {
            'issue': 'Model Architecture Mismatch',
            'probability': '70%',
            'check': '''
# Check if input projection was added:
print(model)  # Look for input_projection layer
''',
            'fix': '''
# Add to model __init__:
self.input_projection = nn.Linear(205, hidden_dim)

# Add to model forward (before GNN):
x = F.relu(self.input_projection(data.x))
'''
        },
        {
            'issue': 'Loss Function Setup',
            'probability': '60%',
            'check': '''
# Check loss function:
print(f"Loss function: {criterion}")
print(f"Loss value: {loss.item()}")
''',
            'fix': '''
# Correct setup:
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target.squeeze().long())
'''
        }
    ]
    
    for idx, issue in enumerate(issues, 1):
        print(f"\n{idx}. {issue['issue']} (Probability: {issue['probability']})")
        print(f"\n   Check:")
        print(issue['check'])
        print(f"\n   Fix:")
        print(issue['fix'])


def create_minimal_test():
    """Create a minimal test to verify everything works"""
    script = '''
"""
Minimal Training Test
Tests one forward/backward pass to catch bugs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GatedGraphConv, global_max_pool

# Create simple model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(205, 200)
        self.ggc = GatedGraphConv(200, num_layers=2)
        self.fc = nn.Linear(200, 2)
    
    def forward(self, data):
        x = F.relu(self.proj(data.x))
        x = F.relu(self.ggc(x, data.edge_index))
        x = global_max_pool(x, data.batch)
        return self.fc(x)

# Create dummy data (2 graphs)
graph1 = Data(
    x=torch.randn(10, 205),
    edge_index=torch.tensor([[0,1,2], [1,2,3]], dtype=torch.long),
    y=torch.tensor([1])
)
graph2 = Data(
    x=torch.randn(15, 205),
    edge_index=torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long),
    y=torch.tensor([0])
)

batch = Batch.from_data_list([graph1, graph2])

# Test training
model = TestModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("="*80)
print("MINIMAL TRAINING TEST")
print("="*80)

for epoch in range(3):
    model.train()
    optimizer.zero_grad()
    
    # Forward
    output = model(batch)
    target = batch.y.squeeze().long()
    
    print(f"\\nEpoch {epoch+1}:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output}")
    print(f"  Target shape: {target.shape}")
    print(f"  Target values: {target}")
    
    # Loss
    loss = criterion(output, target)
    print(f"  Loss: {loss.item():.4f}")
    
    # Accuracy
    pred = output.argmax(dim=1)
    acc = (pred == target).float().mean()
    print(f"  Predictions: {pred}")
    print(f"  Accuracy: {acc.item():.2%}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient norm: {grad_norm:.4f}")
    
    optimizer.step()

print("\\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\\nIf you see:")
print("  - Loss decreasing")
print("  - Gradients flowing (norm > 0)")
print("  - Predictions changing")
print("  - Accuracy improving")
print("\\nThen your setup is correct!")
'''
    
    with open('minimal_training_test.py', 'w', encoding='utf-8') as f:
        f.write(script)
    
    print("\n" + "="*80)
    print("MINIMAL TEST CREATED")
    print("="*80)
    print("\n* Created: minimal_training_test.py")
    print("Run: python minimal_training_test.py")
    print("\nThis will show if basic training works.")


def main():
    print("\n" + "#"*80)
    print("# CONFIGURATION VALIDATOR")
    print("#"*80)
    
    # Check config
    check_config_params()
    
    # Check loss
    check_loss_calculation()
    
    # Analyze README
    analyze_readme_results()
    
    # Diagnose current issue
    diagnose_current_performance()
    
    # Create test
    create_minimal_test()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\n🎯 Action Plan:")
    print("   1. Run: python minimal_training_test.py")
    print("   2. If that works (acc improves), then check your data loading")
    print("   3. If that fails too, check model/loss setup")
    print("   4. Share the output from minimal test")
    print()


if __name__ == "__main__":
    main()