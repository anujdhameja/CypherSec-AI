"""
Final diagnostic to check why accuracy is still very low
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from collections import Counter

from src.utils.objects.input_dataset import InputDataset
from src.process.balanced_training_config import BalancedDevignModel
from torch_geometric.loader import DataLoader
import configs

print("="*80)
print("FINAL DIAGNOSTIC - CHECKING LABELS AND PREDICTIONS")
print("="*80)

# Load data
paths = configs.Paths()
dataset_path = Path(paths.input)
full_dataset = InputDataset(str(dataset_path), max_files=2)

print(f"Loaded {len(full_dataset)} samples")

# Create model
embed = configs.Embed()
model = BalancedDevignModel(
    input_dim=embed.nodes_dim,
    output_dim=2,
    hidden_dim=128,
    num_steps=3,
    dropout=0.2
)

# Create data loader
batch_data = [full_dataset[i] for i in range(min(32, len(full_dataset)))]
loader = DataLoader(batch_data, batch_size=32)

print(f"\n1. CHECKING LABELS:")
all_labels = []
for i in range(len(full_dataset)):
    data = full_dataset[i]
    all_labels.append(data.y.item())

label_counts = Counter(all_labels)
print(f"   Label distribution: {dict(label_counts)}")
print(f"   Label types: {[type(l) for l in all_labels[:5]]}")
print(f"   Label values: {all_labels[:10]}")

print(f"\n2. CHECKING MODEL PREDICTIONS:")
model.eval()
with torch.no_grad():
    for batch in loader:
        print(f"   Batch shape: {batch.x.shape}")
        print(f"   Batch labels: {batch.y}")
        print(f"   Batch label types: {batch.y.dtype}")
        
        # Forward pass
        output = model(batch)
        print(f"   Raw output shape: {output.shape}")
        print(f"   Raw output sample: {output[:5]}")
        
        # Check softmax
        probs = torch.softmax(output, dim=-1)
        print(f"   Softmax probs: {probs[:5]}")
        
        # Check predictions
        preds = output.argmax(dim=-1)
        print(f"   Predictions: {preds}")
        print(f"   True labels: {batch.y.squeeze().long()}")
        
        # Check accuracy manually
        targets = batch.y.squeeze().long()
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        acc = correct / total
        print(f"   Manual accuracy: {acc:.4f} ({correct}/{total})")
        
        break

print(f"\n3. CHECKING LOSS CALCULATION:")
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in loader:
        output = model(batch)
        targets = batch.y.squeeze().long()
        
        print(f"   Output shape: {output.shape}")
        print(f"   Targets shape: {targets.shape}")
        print(f"   Output dtype: {output.dtype}")
        print(f"   Targets dtype: {targets.dtype}")
        
        try:
            loss = criterion(output, targets)
            print(f"   ✓ Loss calculation successful: {loss.item():.6f}")
        except Exception as e:
            print(f"   ❌ Loss calculation failed: {e}")
        
        break

print(f"\n4. CHECKING IF MODEL IS STUCK:")
# Check if model always predicts the same class
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for i in range(min(100, len(full_dataset))):
        data = full_dataset[i]
        if not hasattr(data, 'batch'):
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        
        output = model(data)
        pred = output.argmax(dim=-1).item()
        true_label = data.y.item()
        
        all_predictions.append(pred)
        all_true_labels.append(true_label)

pred_counts = Counter(all_predictions)
true_counts = Counter(all_true_labels)

print(f"   Prediction distribution: {dict(pred_counts)}")
print(f"   True label distribution: {dict(true_counts)}")

if len(set(all_predictions)) == 1:
    print(f"   ❌ MODEL IS STUCK: Always predicts class {list(set(all_predictions))[0]}")
    stuck_class = list(set(all_predictions))[0]
    expected_acc = true_counts[stuck_class] / len(all_true_labels)
    print(f"   Expected accuracy if stuck: {expected_acc:.4f} ({expected_acc*100:.2f}%)")
else:
    print(f"   ✓ Model predicts both classes")

print(f"\n5. CHECKING FEATURE QUALITY:")
# Check if features are meaningful
sample_data = full_dataset[0]
print(f"   Feature shape: {sample_data.x.shape}")
print(f"   Feature range: [{sample_data.x.min():.6f}, {sample_data.x.max():.6f}]")
print(f"   Feature mean: {sample_data.x.mean():.6f}")
print(f"   Feature std: {sample_data.x.std():.6f}")
print(f"   Non-zero features: {(sample_data.x != 0).sum().item()}/{sample_data.x.numel()}")

# Check feature diversity
feature_means = []
for i in range(min(10, len(full_dataset))):
    data = full_dataset[i]
    feature_means.append(data.x.mean().item())

print(f"   Feature means across samples: {feature_means}")
print(f"   Feature diversity (std of means): {np.std(feature_means):.6f}")

print(f"\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)