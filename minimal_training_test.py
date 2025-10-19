
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
    
    print(f"\nEpoch {epoch+1}:")
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

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nIf you see:")
print("  - Loss decreasing")
print("  - Gradients flowing (norm > 0)")
print("  - Predictions changing")
print("  - Accuracy improving")
print("\nThen your setup is correct!")
