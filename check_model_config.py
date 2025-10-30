#!/usr/bin/env python3
"""
Check the actual configuration of the saved model
"""

import torch

def check_model_config():
    model_path = "models/production_model_config9_v1.0.pth"
    
    print("="*80)
    print("CHECKING SAVED MODEL CONFIGURATION")
    print("="*80)
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        print(f"Model file: {model_path}")
        print(f"Number of parameters: {len(state_dict)}")
        
        print("\nParameter shapes:")
        print("-" * 50)
        
        for name, tensor in state_dict.items():
            print(f"{name:25s}: {list(tensor.shape)}")
        
        # Infer configuration from shapes
        print("\n" + "="*80)
        print("INFERRED CONFIGURATION")
        print("="*80)
        
        if 'input_proj.weight' in state_dict:
            input_dim = state_dict['input_proj.weight'].shape[1]
            hidden_dim = state_dict['input_proj.weight'].shape[0]
            print(f"Input dimension: {input_dim}")
            print(f"Hidden dimension: {hidden_dim}")
        
        if 'ggc.weight' in state_dict:
            num_steps = state_dict['ggc.weight'].shape[0]
            print(f"GNN steps: {num_steps}")
        
        if 'fc1.weight' in state_dict:
            fc1_input = state_dict['fc1.weight'].shape[1]
            fc1_output = state_dict['fc1.weight'].shape[0]
            print(f"FC1: {fc1_input} → {fc1_output}")
            
            # Check if dual pooling
            if fc1_input == hidden_dim * 2:
                print("Pooling: mean_max (dual pooling)")
            elif fc1_input == hidden_dim:
                print("Pooling: single (mean or max)")
            else:
                print(f"Pooling: unknown (input={fc1_input}, hidden={hidden_dim})")
        
        if 'fc3.weight' in state_dict:
            output_dim = state_dict['fc3.weight'].shape[0]
            print(f"Output dimension: {output_dim}")
        
        print("\n" + "="*80)
        print("CORRECT CONFIGURATION FOR LOADING")
        print("="*80)
        print("Use these parameters:")
        print(f"  input_dim = {input_dim}")
        print(f"  hidden_dim = {hidden_dim}")
        print(f"  num_steps = {num_steps}")
        print(f"  output_dim = {output_dim}")
        print(f"  pooling = mean_max")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model_config()