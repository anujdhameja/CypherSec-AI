"""
Debug the emergency model to see what's going wrong
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from collections import Counter

from src.utils.objects.input_dataset import InputDataset
from src.process.balanced_training_config import BalancedDevignModel
import configs

print("="*80)
print("DEBUGGING EMERGENCY MODEL")
print("="*80)

# Load data
paths = configs.Paths()
dataset_path = Path(paths.input)
full_dataset = InputDataset(str(dataset_path), max_files=2)  # Just 2 files for debugging

print(f"\n1. DATA INSPECTION:")
print(f"   Total samples: {len(full_dataset)}")

# Check first few samples
for i in range(min(3, len(full_dataset))):
    data = full_dataset[i]
    print(f"\n   Sample {i}:")
    print(f"   - Type: {type(data)}")
    print(f"   - Has x: {hasattr(data, 'x')}")
    print(f"   - Has edge_index: {hasattr(data, 'edge_index')}")
    print(f"   - Has y: {hasattr(data, 'y')}")
    print(f"   - Has batch: {hasattr(data, 'batch')}")
    
    if hasattr(data, 'x'):
        print(f"   - x shape: {data.x.shape}")
        print(f"   - x dtype: {data.x.dtype}")
        print(f"   - x range: [{data.x.min():.3f}, {data.x.max():.3f}]")
        print(f"   - x has NaN: {torch.isnan(data.x).any()}")
        print(f"   - x has Inf: {torch.isinf(data.x).any()}")
    
    if hasattr(data, 'edge_index'):
        print(f"   - edge_index shape: {data.edge_index.shape}")
        print(f"   - edge_index dtype: {data.edge_index.dtype}")
        print(f"   - edge_index range: [{data.edge_index.min()}, {data.edge_index.max()}]")
    
    if hasattr(data, 'y'):
        print(f"   - y: {data.y}")
        print(f"   - y dtype: {data.y.dtype}")

# Check label distribution
all_labels = []
for i in range(len(full_dataset)):
    data = full_dataset[i]
    if hasattr(data, 'y'):
        all_labels.append(data.y.item())

label_counts = Counter(all_labels)
print(f"\n2. LABEL DISTRIBUTION:")
print(f"   {dict(label_counts)}")
print(f"   Unique labels: {set(all_labels)}")

# Test model creation
print(f"\n3. MODEL TEST:")
embed = configs.Embed()
print(f"   Embed nodes_dim: {embed.nodes_dim}")

model = BalancedDevignModel(
    input_dim=embed.nodes_dim,
    output_dim=2,
    hidden_dim=128,
    num_steps=3,
    dropout=0.2
)

print(f"   Model created successfully")

# Test forward pass with single sample
print(f"\n4. FORWARD PASS TEST:")
model.eval()
with torch.no_grad():
    data = full_dataset[0]
    
    # Add batch dimension if missing
    if not hasattr(data, 'batch'):
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    
    print(f"   Input x shape: {data.x.shape}")
    print(f"   Input edge_index shape: {data.edge_index.shape}")
    print(f"   Input batch shape: {data.batch.shape}")
    
    try:
        output = model(data)
        print(f"   ✓ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output values: {output}")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Check softmax
        probs = torch.softmax(output, dim=-1)
        print(f"   Softmax probs: {probs}")
        
        # Check prediction
        pred = output.argmax(dim=-1)
        print(f"   Prediction: {pred.item()}")
        print(f"   True label: {data.y.item()}")
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

# Test batch processing
print(f"\n5. BATCH PROCESSING TEST:")
from torch_geometric.loader import DataLoader

try:
    # Create small batch
    batch_data = [full_dataset[i] for i in range(min(4, len(full_dataset)))]
    loader = DataLoader(batch_data, batch_size=4)
    
    for batch in loader:
        print(f"   Batch x shape: {batch.x.shape}")
        print(f"   Batch edge_index shape: {batch.edge_index.shape}")
        print(f"   Batch y shape: {batch.y.shape}")
        print(f"   Batch num_graphs: {batch.num_graphs}")
        
        with torch.no_grad():
            output = model(batch)
            print(f"   ✓ Batch forward pass successful")
            print(f"   Batch output shape: {output.shape}")
            print(f"   Batch predictions: {output.argmax(dim=-1)}")
            print(f"   Batch true labels: {batch.y}")
        break
        
except Exception as e:
    print(f"   ❌ Batch processing failed: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)