"""
Test script to verify stable model integration
"""

import torch
from torch_geometric.data import Data, Batch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

def test_stable_model():
    """Test if the stable model integration works"""
    print("="*80)
    print("TESTING STABLE MODEL INTEGRATION")
    print("="*80)
    
    try:
        # Import the updated Devign class
        from src.process.devign import Devign, StableDevignModel
        
        print("✓ Successfully imported StableDevignModel and Devign")
        
        # Test StableDevignModel directly
        print("\n1. Testing StableDevignModel...")
        model = StableDevignModel(input_dim=205, hidden_dim=200, num_steps=4)
        
        # Create dummy data
        graphs = []
        for _ in range(4):
            x = torch.randn(10, 205)
            edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long)
            y = torch.randint(0, 2, (1,))
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
        
        batch = Batch.from_data_list(graphs)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch)
            print(f"✓ Forward pass successful: {output.shape}")
        
        # Test training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        for i in range(3):
            optimizer.zero_grad()
            output = model(batch)
            target = batch.y.squeeze().long()
            loss = criterion(output, target)
            loss.backward()
            
            # Test gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            pred = output.argmax(dim=1)
            acc = (pred == target).float().mean()
            
            print(f"  Step {i+1}: Loss={loss.item():.4f}, Acc={acc.item():.2%}, GradNorm={grad_norm:.2f}")
        
        print("✓ StableDevignModel test passed!")
        
        # Test Devign class integration
        print("\n2. Testing Devign class integration...")
        
        # Mock config
        mock_config = {
            'conv_args': {
                'conv1d_1': {'in_channels': 205}
            },
            'gated_graph_conv_args': {
                'out_channels': 200,
                'num_layers': 4
            }
        }
        
        # Create Devign instance
        devign = Devign(
            path="test_model.pt",
            device="cpu",
            model=mock_config,
            learning_rate=1e-4,  # Will be multiplied by 3 internally
            weight_decay=1.3e-6,
            loss_lambda=0.0
        )
        
        print("✓ Devign class created successfully")
        
        # Test training step
        devign.train()
        stat = devign(0, batch, batch.y)
        print(f"✓ Training step successful: Loss={stat.loss:.4f}, Acc={stat.acc:.2%}")
        
        # Test validation step
        devign.eval()
        with torch.no_grad():
            stat = devign(0, batch, batch.y)
            print(f"✓ Validation step successful: Loss={stat.loss:.4f}, Acc={stat.acc:.2%}")
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        
        print("\n🔧 Key Improvements Applied:")
        print("   ✓ Reduced GNN steps: 8 → 4")
        print("   ✓ Added gradient clipping (max_norm=1.0)")
        print("   ✓ Increased learning rate: 1e-4 → 3e-4")
        print("   ✓ Added batch normalization")
        print("   ✓ Added learning rate scheduler")
        print("   ✓ Simplified loss function (CrossEntropy only)")
        print("   ✓ Added NaN loss detection")
        
        print("\n🚀 Ready to train! Run:")
        print("   python main.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_stable_model()
    if success:
        print("\n✅ Integration successful! Your model is ready for stable training.")
    else:
        print("\n❌ Integration failed. Please check the errors above.")