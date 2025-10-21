import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader
from src.process.balanced_training_config import BalancedDevignModel
import configs

print("="*80)
print("OVERFIT SINGLE BATCH TEST")
print("="*80)
print("If model can't overfit 1 batch → Architecture is broken")
print("If model CAN overfit 1 batch → Problem is elsewhere\n")

# Load single batch
input_path = Path('data/input')
files = sorted(input_path.glob('*_cpg_input.pkl'))[:1]
df = pd.read_pickle(files[0])
graphs = df['input'].tolist()[:16]  # Just 16 samples

print(f"Using {len(graphs)} samples")
labels = [g.y.item() for g in graphs]
print(f"Labels: {labels}")
print(f"Distribution: {labels.count(0.0)} zeros, {labels.count(1.0)} ones\n")

# Create model
device = torch.device('cpu')
model = BalancedDevignModel(
    input_dim=100,
    output_dim=2,
    hidden_dim=128,
    num_steps=3,
    dropout=0.2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training for 100 epochs on same batch...")
print("Target: Should reach 100% accuracy\n")

# Create a simple batch manually
from torch_geometric.data import Batch

# Combine all graphs into a single batch
batch = Batch.from_data_list(graphs).to(device)

for epoch in range(100):
    model.train()
    
    optimizer.zero_grad()
    output = model(batch)
    target = batch.y.long()
    
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    # Check accuracy every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            output = model(batch)
            pred = output.argmax(dim=1)
            acc = (pred == batch.y.long()).sum().item() / len(batch.y) * 100
            
            print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, Acc={acc:.1f}%")

print("\n" + "="*80)
print("RESULT")
print("="*80)

# Final evaluation
model.eval()
with torch.no_grad():
    for batch in DataLoader(graphs, batch_size=16, shuffle=False):
        batch = batch.to(device)
        output = model(batch)
        pred = output.argmax(dim=1)
        target = batch.y.long()
        acc = (pred == target).sum().item() / len(target) * 100
        
        print(f"\nFinal accuracy: {acc:.1f}%")
        print(f"Final loss: {loss.item():.4f}")
        
        # Show predictions vs targets
        print(f"\nPredictions: {pred.tolist()}")
        print(f"Targets:     {target.tolist()}")
        print(f"Matches:     {(pred == target).tolist()}")

if acc >= 95:
    print("\n✓ SUCCESS: Model can overfit → Architecture works")
    print("  Problem is likely: data quality, shuffling, or training process")
elif acc >= 70:
    print("\n⚠️ PARTIAL: Model shows some learning → Architecture might be okay")
    print("  But optimization might be suboptimal")
else:
    print("\n✗ FAILURE: Model cannot overfit → Architecture is broken")
    print("  Need to fix: pooling, layers, or forward pass")

# Additional diagnostics
print(f"\n" + "="*80)
print("DIAGNOSTIC INFO")
print("="*80)

with torch.no_grad():
    for batch in DataLoader(graphs, batch_size=16, shuffle=False):
        batch = batch.to(device)
        output = model(batch)
        
        print(f"Raw output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"Raw output mean: {output.mean().item():.4f}")
        print(f"Raw output std: {output.std().item():.4f}")
        
        # Check if outputs are reasonable
        if torch.isnan(output).any():
            print("⚠️ WARNING: NaN values in output!")
        if torch.isinf(output).any():
            print("⚠️ WARNING: Infinite values in output!")
        if output.std().item() < 0.01:
            print("⚠️ WARNING: Very low output variance - model might be stuck!")
        
        # Show softmax probabilities for first few samples
        probs = torch.softmax(output, dim=1)
        print(f"\nFirst 5 softmax probabilities:")
        for i in range(min(5, len(probs))):
            print(f"  Sample {i}: [{probs[i][0]:.3f}, {probs[i][1]:.3f}] → pred={pred[i].item()}, target={target[i].item()}")
        
        break