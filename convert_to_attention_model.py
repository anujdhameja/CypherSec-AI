#!/usr/bin/env python3
"""
Convert Trained Model to Attention-Enhanced Version

This script converts the existing final_model.pth to an attention-enhanced version
that can provide explainable predictions by showing which nodes the model
pays attention to.

Usage:
    python convert_to_attention_model.py
    
Output:
    - models/final_model_with_attention.pth (attention-enhanced model)
    - Test predictions with attention weights
"""

import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append('src')

from src.process.attention_devign_model import AttentionDevignModel
from src.process.balanced_training_config import BalancedDevignModel
from torch_geometric.data import Data


def main():
    print("="*80)
    print("CONVERT MODEL TO ATTENTION-ENHANCED VERSION")
    print("="*80)
    
    # Paths
    original_model_path = "models/final_model.pth"
    attention_model_path = "models/final_model_with_attention.pth"
    
    print(f"Source model: {original_model_path}")
    print(f"Target model: {attention_model_path}")
    
    # Check if original model exists
    if not os.path.exists(original_model_path):
        print(f"❌ Original model not found: {original_model_path}")
        print(f"💡 Make sure you have trained the model first")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Step 1: Load original model to verify architecture
        print(f"\n🔍 Verifying original model...")
        original_model = BalancedDevignModel(
            input_dim=100,
            output_dim=2,
            hidden_dim=256,
            num_steps=5,
            dropout=0.2
        )
        
        original_state_dict = torch.load(original_model_path, map_location='cpu')
        original_model.load_state_dict(original_state_dict)
        print(f"✅ Original model loaded successfully")
        
        # Step 2: Create attention-enhanced model
        print(f"\n🔧 Creating attention-enhanced model...")
        attention_model = AttentionDevignModel(
            input_dim=100,
            output_dim=2,
            hidden_dim=256,
            num_steps=5,
            dropout=0.2,
            pooling='mean_max'
        )
        
        # Step 3: Transfer compatible weights
        print(f"\n🔄 Transferring weights...")
        attention_state_dict = attention_model.state_dict()
        transferred_keys = []
        skipped_keys = []
        
        for key in original_state_dict:
            if key in attention_state_dict:
                if original_state_dict[key].shape == attention_state_dict[key].shape:
                    attention_state_dict[key] = original_state_dict[key]
                    transferred_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (not found)")
        
        # Load transferred weights
        attention_model.load_state_dict(attention_state_dict)
        
        print(f"✅ Transferred {len(transferred_keys)} weight tensors")
        print(f"⚠️  Skipped {len(skipped_keys)} incompatible tensors")
        
        if transferred_keys:
            print(f"\n📝 Transferred weights:")
            for key in transferred_keys:
                print(f"   ✓ {key}")
        
        if skipped_keys:
            print(f"\n📝 Skipped weights (will use random initialization):")
            for key in skipped_keys:
                print(f"   ⚠️  {key}")
        
        # Step 4: Save attention model
        print(f"\n💾 Saving attention-enhanced model...")
        torch.save(attention_model.state_dict(), attention_model_path)
        print(f"✅ Attention model saved to: {attention_model_path}")
        
        # Step 5: Test both models
        print(f"\n🧪 Testing both models...")
        test_models_comparison(original_model, attention_model)
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"📄 You can now use the attention model with:")
        print(f"   from src.inference.explainable_predictor import ExplainablePredictor")
        print(f"   predictor = ExplainablePredictor('{attention_model_path}')")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


def test_models_comparison(original_model, attention_model):
    """Test and compare original vs attention models"""
    
    print(f"\n🔬 Comparing model predictions...")
    
    # Create test data
    num_nodes = 10
    x = torch.randn(num_nodes, 100)
    
    edges = []
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    edges.extend([[0, 5], [5, 0], [2, 7], [7, 2]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test original model
    original_model.eval()
    with torch.no_grad():
        original_output = original_model(data)
        original_probs = torch.softmax(original_output, dim=1)
        original_pred = original_output.argmax(dim=1).item()
        original_conf = original_probs[0][original_pred].item()
    
    # Test attention model
    attention_model.eval()
    with torch.no_grad():
        attention_output, attention_weights = attention_model(data, return_attention=True)
        attention_probs = torch.softmax(attention_output, dim=1)
        attention_pred = attention_output.argmax(dim=1).item()
        attention_conf = attention_probs[0][attention_pred].item()
    
    # Compare results
    print(f"\n📊 Model Comparison Results:")
    print(f"   Original Model:")
    print(f"     Prediction: {'Vulnerable' if original_pred == 1 else 'Safe'}")
    print(f"     Confidence: {original_conf:.2%}")
    print(f"     Raw output: [{original_output[0][0]:.3f}, {original_output[0][1]:.3f}]")
    
    print(f"\n   Attention Model:")
    print(f"     Prediction: {'Vulnerable' if attention_pred == 1 else 'Safe'}")
    print(f"     Confidence: {attention_conf:.2%}")
    print(f"     Raw output: [{attention_output[0][0]:.3f}, {attention_output[0][1]:.3f}]")
    print(f"     Attention range: [{attention_weights.min():.3f}, {attention_weights.max():.3f}]")
    print(f"     High attention nodes: {(attention_weights > 0.7).sum().item()}")
    
    # Check if predictions are similar
    pred_match = original_pred == attention_pred
    conf_diff = abs(original_conf - attention_conf)
    
    print(f"\n✅ Compatibility Check:")
    print(f"   Predictions match: {'✓' if pred_match else '✗'}")
    print(f"   Confidence difference: {conf_diff:.3f}")
    
    if pred_match and conf_diff < 0.1:
        print(f"   🎉 Models are highly compatible!")
    elif pred_match:
        print(f"   ✅ Models agree on prediction")
    else:
        print(f"   ⚠️  Models disagree - this is expected due to architectural changes")
    
    # Show attention details
    print(f"\n🎯 Attention Analysis:")
    attention_np = attention_weights.cpu().numpy()
    sorted_indices = np.argsort(attention_np)[::-1]
    
    print(f"   Top 5 most important nodes:")
    for i in range(min(5, len(sorted_indices))):
        node_id = sorted_indices[i]
        attention_score = attention_np[node_id]
        print(f"     Node {node_id}: {attention_score:.3f}")


def verify_attention_model():
    """Verify the converted attention model works correctly"""
    
    attention_model_path = "models/final_model_with_attention.pth"
    
    if not os.path.exists(attention_model_path):
        print(f"❌ Attention model not found: {attention_model_path}")
        return False
    
    try:
        # Load and test attention model
        model = AttentionDevignModel(
            input_dim=100,
            output_dim=2,
            hidden_dim=256,
            num_steps=5,
            dropout=0.2
        )
        
        state_dict = torch.load(attention_model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        # Test with dummy data
        from torch_geometric.data import Data
        
        x = torch.randn(8, 100)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        batch = torch.zeros(8, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        
        with torch.no_grad():
            output, attention = model(data, return_attention=True)
            
        print(f"✅ Attention model verification successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Attention shape: {attention.shape}")
        print(f"   Attention range: [{attention.min():.3f}, {attention.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Attention model verification failed: {e}")
        return False


if __name__ == "__main__":
    main()
    
    # Verify the converted model
    print(f"\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    verify_attention_model()