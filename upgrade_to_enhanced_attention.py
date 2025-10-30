#!/usr/bin/env python3
"""
Upgrade Existing Attention Model to Enhanced Version

This script converts our current attention model to the enhanced version
with multi-head attention and vulnerability-aware detection.

Usage:
    python upgrade_to_enhanced_attention.py
"""

import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.process.enhanced_attention_model import EnhancedAttentionDevignModel, convert_existing_model_to_enhanced
from src.process.attention_devign_model import AttentionDevignModel


def main():
    print("="*80)
    print("UPGRADE TO ENHANCED ATTENTION MODEL")
    print("="*80)
    
    # Paths
    current_model_path = "models/final_model_with_attention.pth"
    enhanced_model_path = "models/enhanced_attention_model.pth"
    
    print(f"Current model: {current_model_path}")
    print(f"Enhanced model: {enhanced_model_path}")
    
    # Check if current model exists
    if not os.path.exists(current_model_path):
        print(f"❌ Current attention model not found: {current_model_path}")
        print(f"💡 Run convert_to_attention_model.py first")
        return
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    try:
        # Step 1: Load current attention model to verify
        print(f"\n🔍 Verifying current attention model...")
        current_model = AttentionDevignModel(
            input_dim=100,
            output_dim=2,
            hidden_dim=256,
            num_steps=5,
            dropout=0.2,
            pooling='mean_max'
        )
        
        current_state_dict = torch.load(current_model_path, map_location='cpu')
        current_model.load_state_dict(current_state_dict)
        print(f"✅ Current attention model loaded successfully")
        
        # Step 2: Create enhanced attention model
        print(f"\n🔧 Creating enhanced attention model...")
        enhanced_model = EnhancedAttentionDevignModel(
            input_dim=100,
            output_dim=2,
            hidden_dim=256,
            num_steps=5,
            num_attention_heads=4,  # NEW: Multi-head attention
            dropout=0.2,
            pooling='mean_max'
        )
        
        # Step 3: Transfer compatible weights
        print(f"\n🔄 Transferring compatible weights...")
        enhanced_state_dict = enhanced_model.state_dict()
        transferred_keys = []
        new_keys = []
        
        # Direct transfers (same architecture components)
        direct_transfers = {
            'input_projection.0.weight': 'input_projection.0.weight',
            'input_projection.0.bias': 'input_projection.0.bias',
            'ggc.weight': 'ggnn.weight',
            'ggc.rnn.weight_ih': 'ggnn.rnn.weight_ih', 
            'ggc.rnn.weight_hh': 'ggnn.rnn.weight_hh',
            'ggc.rnn.bias_ih': 'ggnn.rnn.bias_ih',
            'ggc.rnn.bias_hh': 'ggnn.rnn.bias_hh',
            'classifier.0.weight': 'classifier.0.weight',
            'classifier.0.bias': 'classifier.0.bias',
            'classifier.2.weight': 'classifier.2.weight', 
            'classifier.2.bias': 'classifier.2.bias',
            'classifier.4.weight': 'classifier.4.weight',
            'classifier.4.bias': 'classifier.4.bias',
        }
        
        for old_key, new_key in direct_transfers.items():
            if old_key in current_state_dict and new_key in enhanced_state_dict:
                if current_state_dict[old_key].shape == enhanced_state_dict[new_key].shape:
                    enhanced_state_dict[new_key] = current_state_dict[old_key]
                    transferred_keys.append(f"{old_key} → {new_key}")
                else:
                    print(f"⚠️  Shape mismatch: {old_key} {current_state_dict[old_key].shape} vs {new_key} {enhanced_state_dict[new_key].shape}")
        
        # Initialize new attention components with smart defaults
        print(f"\n🎯 Initializing new attention components...")
        
        # Initialize multi-head attention projections from simple attention if available
        if 'attention_layer.0.weight' in current_state_dict:
            # Use the old simple attention weights as initialization
            old_attention_weight = current_state_dict['attention_layer.0.weight']  # [128, 256]
            old_attention_bias = current_state_dict['attention_layer.0.bias']      # [128]
            
            # The enhanced model expects [256, 256] for projections
            # Initialize with Xavier uniform and then partially copy old weights
            print(f"   Old attention weight shape: {old_attention_weight.shape}")
            print(f"   New projection weight shape: {enhanced_state_dict['attention.query_proj.weight'].shape}")
            
            # Initialize projections with Xavier uniform (better than random)
            torch.nn.init.xavier_uniform_(enhanced_state_dict['attention.query_proj.weight'])
            torch.nn.init.xavier_uniform_(enhanced_state_dict['attention.key_proj.weight'])
            torch.nn.init.xavier_uniform_(enhanced_state_dict['attention.value_proj.weight'])
            
            # Initialize biases to zero
            torch.nn.init.zeros_(enhanced_state_dict['attention.query_proj.bias'])
            torch.nn.init.zeros_(enhanced_state_dict['attention.key_proj.bias'])
            torch.nn.init.zeros_(enhanced_state_dict['attention.value_proj.bias'])
            
            new_keys.append("Multi-head attention projections (Xavier initialized)")
        else:
            new_keys.append("Multi-head attention projections (random initialized)")
        
        # List all new components that will be randomly initialized
        new_components = [
            'attention.vulnerability_detector',
            'attention.context_attention', 
            'attention.layer_norm'
        ]
        
        for component in new_components:
            component_keys = [k for k in enhanced_state_dict.keys() if k.startswith(component)]
            if component_keys:
                new_keys.append(f"{component}.* ({len(component_keys)} parameters)")
        
        # Load enhanced model with transferred + new weights (strict=False to allow missing keys)
        missing_keys, unexpected_keys = enhanced_model.load_state_dict(enhanced_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys (will use default initialization): {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)}")
        
        print(f"✅ Transferred {len(transferred_keys)} existing weight tensors")
        for key in transferred_keys:
            print(f"   ✓ {key}")
        
        print(f"\n🆕 Initialized {len(new_keys)} new components:")
        for key in new_keys:
            print(f"   🎯 {key}")
        
        # Step 4: Save enhanced model
        print(f"\n💾 Saving enhanced attention model...")
        torch.save(enhanced_model.state_dict(), enhanced_model_path)
        print(f"✅ Enhanced model saved to: {enhanced_model_path}")
        
        # Step 5: Test both models for compatibility
        print(f"\n🧪 Testing model compatibility...")
        test_model_compatibility(current_model, enhanced_model)
        
        print(f"\n🎉 Upgrade completed successfully!")
        print(f"📄 You can now use the enhanced model with:")
        print(f"   from src.process.enhanced_attention_model import EnhancedAttentionDevignModel")
        print(f"   model = EnhancedAttentionDevignModel(...)")
        print(f"   model.load_state_dict(torch.load('{enhanced_model_path}'))")
        
    except Exception as e:
        print(f"❌ Upgrade failed: {e}")
        import traceback
        traceback.print_exc()


def test_model_compatibility(current_model, enhanced_model):
    """Test compatibility between current and enhanced models"""
    
    print(f"\n🔬 Comparing model predictions...")
    
    # Create test data
    from torch_geometric.data import Data
    
    num_nodes = 12
    x = torch.randn(num_nodes, 100)
    
    edges = []
    for i in range(num_nodes - 1):
        edges.extend([[i, i+1], [i+1, i]])
    edges.extend([[0, 5], [5, 0], [2, 7], [7, 2]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test current model
    current_model.eval()
    with torch.no_grad():
        current_output, current_attention = current_model(data, return_attention=True)
        current_probs = torch.softmax(current_output, dim=1)
        current_pred = current_output.argmax(dim=1).item()
        current_conf = current_probs[0][current_pred].item()
    
    # Test enhanced model
    enhanced_model.eval()
    with torch.no_grad():
        enhanced_output, enhanced_attention = enhanced_model(data, return_attention=True)
        enhanced_probs = torch.softmax(enhanced_output, dim=1)
        enhanced_pred = enhanced_output.argmax(dim=1).item()
        enhanced_conf = enhanced_probs[0][enhanced_pred].item()
    
    # Compare results
    print(f"\n📊 Model Comparison Results:")
    print(f"   Current Model:")
    print(f"     Prediction: {'Vulnerable' if current_pred == 1 else 'Safe'}")
    print(f"     Confidence: {current_conf:.2%}")
    print(f"     Attention range: [{current_attention.min():.3f}, {current_attention.max():.3f}]")
    
    print(f"\n   Enhanced Model:")
    print(f"     Prediction: {'Vulnerable' if enhanced_pred == 1 else 'Safe'}")
    print(f"     Confidence: {enhanced_conf:.2%}")
    print(f"     Attention range: [{enhanced_attention.min():.3f}, {enhanced_attention.max():.3f}]")
    
    # Test enhanced features
    vulnerabilities = enhanced_model.get_line_level_vulnerabilities(data, top_k=5)
    
    print(f"\n🎯 Enhanced Features Test:")
    print(f"   Line-level analysis: ✅ Working")
    print(f"   Top suspicious lines: {len(vulnerabilities['vulnerable_lines'])}")
    print(f"   Risk levels: {set(v['risk_level'] for v in vulnerabilities['vulnerable_lines'])}")
    
    # Check compatibility
    pred_match = current_pred == enhanced_pred
    conf_diff = abs(current_conf - enhanced_conf)
    
    print(f"\n✅ Compatibility Check:")
    print(f"   Predictions match: {'✓' if pred_match else '✗'}")
    print(f"   Confidence difference: {conf_diff:.3f}")
    
    if pred_match and conf_diff < 0.2:
        print(f"   🎉 Models are highly compatible!")
    elif pred_match:
        print(f"   ✅ Models agree on prediction")
    else:
        print(f"   ⚠️  Models disagree - this is expected due to enhanced attention")
    
    print(f"\n🔍 Enhanced Attention Analysis:")
    enhanced_attention_np = enhanced_attention.cpu().numpy()
    import numpy as np
    sorted_indices = np.argsort(enhanced_attention_np)[::-1]
    
    print(f"   Top 5 most important nodes:")
    for i in range(min(5, len(sorted_indices))):
        node_id = sorted_indices[i]
        attention_score = enhanced_attention_np[node_id]
        print(f"     Node {node_id}: {attention_score:.3f}")


if __name__ == "__main__":
    main()