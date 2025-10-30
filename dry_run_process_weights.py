#!/usr/bin/env python3
"""
Dry Run Script for -p (Process) Step
Shows current weight values and model configuration without training
"""

import torch
import torch.nn as nn
import configs
from src.process.devign import Devign

def dry_run_process_step():
    """
    Dry run the -p step to show current weight values and configuration
    """
    print("=" * 80)
    print("DRY RUN: PROCESS STEP (-p) WEIGHT ANALYSIS")
    print("=" * 80)
    
    # Load configurations (same as main.py process_task)
    context = configs.Process()
    devign_config = configs.Devign()
    
    print("\n📋 CONFIGURATION VALUES:")
    print("-" * 50)
    print(f"Learning Rate: {devign_config.learning_rate}")
    print(f"Weight Decay: {devign_config.weight_decay}")
    print(f"Loss Lambda: {devign_config.loss_lambda}")
    
    print(f"\nModel Configuration:")
    print(f"  Input Channels: {devign_config.model['conv_args']['conv1d_1']['in_channels']}")
    print(f"  GNN Hidden Dim: {devign_config.model['gated_graph_conv_args']['out_channels']}")
    print(f"  GNN Layers: {devign_config.model['gated_graph_conv_args']['num_layers']}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {context.epochs}")
    print(f"  Batch Size: {context.batch_size}")
    print(f"  Patience: {context.patience}")
    
    # Initialize model (same as main.py)
    PATHS = configs.Paths()
    FILES = configs.Files()
    DEVICE = FILES.get_device()
    
    model_path = PATHS.model + FILES.model
    print(f"\nModel Path: {model_path}")
    print(f"Device: {DEVICE}")
    
    # Create Devign model instance
    print("\n🔧 INITIALIZING MODEL...")
    print("-" * 50)
    
    model = Devign(
        path=model_path, 
        device=DEVICE, 
        model=devign_config.model, 
        learning_rate=devign_config.learning_rate,
        weight_decay=devign_config.weight_decay,
        loss_lambda=devign_config.loss_lambda
    )
    
    print("\n🔍 WEIGHT ANALYSIS:")
    print("-" * 50)
    
    # Analyze model weights
    total_params = 0
    trainable_params = 0
    
    print("\nLayer-by-layer weight analysis:")
    for name, param in model.model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        
        # Show weight statistics
        if param.requires_grad:
            weight_mean = param.data.mean().item()
            weight_std = param.data.std().item()
            weight_min = param.data.min().item()
            weight_max = param.data.max().item()
            
            print(f"\n  {name}:")
            print(f"    Shape: {list(param.shape)}")
            print(f"    Mean: {weight_mean:.6f}")
            print(f"    Std:  {weight_std:.6f}")
            print(f"    Min:  {weight_min:.6f}")
            print(f"    Max:  {weight_max:.6f}")
            print(f"    Params: {param.numel():,}")
    
    print(f"\n📊 PARAMETER SUMMARY:")
    print("-" * 50)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    
    # Check for pre-trained weights
    print(f"\n🔍 PRE-TRAINED MODEL CHECK:")
    print("-" * 50)
    
    import os
    production_model_path = "models/production_model_config9_v1.0.pth"
    if os.path.exists(production_model_path):
        print(f"✅ Pre-trained model found: {production_model_path}")
        try:
            pretrained_state = torch.load(production_model_path, map_location=DEVICE)
            print(f"✅ Pre-trained model loaded successfully")
            print(f"   Keys in state dict: {len(pretrained_state.keys())}")
            
            # Compare with current model
            current_keys = set(model.model.state_dict().keys())
            pretrained_keys = set(pretrained_state.keys())
            
            if current_keys == pretrained_keys:
                print("✅ Model architecture matches pre-trained weights")
            else:
                missing = current_keys - pretrained_keys
                extra = pretrained_keys - current_keys
                if missing:
                    print(f"⚠️  Missing keys: {missing}")
                if extra:
                    print(f"⚠️  Extra keys: {extra}")
        except Exception as e:
            print(f"❌ Error loading pre-trained model: {e}")
    else:
        print(f"ℹ️  No pre-trained model found at: {production_model_path}")
        print("   Model will use random initialization")
    
    # Check optimizer configuration
    print(f"\n⚙️  OPTIMIZER CONFIGURATION:")
    print("-" * 50)
    print(f"Optimizer Type: {type(model.optimizer).__name__}")
    print(f"Learning Rate: {model.optimizer.param_groups[0]['lr']}")
    print(f"Weight Decay: {model.optimizer.param_groups[0].get('weight_decay', 'N/A')}")
    
    if hasattr(model, 'scheduler'):
        print(f"Scheduler: {type(model.scheduler).__name__}")
        print(f"Scheduler Mode: {model.scheduler.mode}")
        print(f"Scheduler Factor: {model.scheduler.factor}")
        print(f"Scheduler Patience: {model.scheduler.patience}")
    
    # Loss function
    print(f"\n📉 LOSS FUNCTION:")
    print("-" * 50)
    if hasattr(model, 'loss_function'):
        print(f"Loss Function: {type(model.loss_function).__name__}")
    elif hasattr(model, 'loss_fn'):
        print(f"Loss Function: {type(model.loss_fn).__name__}")
    else:
        print("Loss Function: CrossEntropyLoss (from Step class)")
    
    # Show weight initialization patterns
    print(f"\n🎲 WEIGHT INITIALIZATION PATTERNS:")
    print("-" * 50)
    
    # Check if weights follow expected initialization patterns
    for name, param in model.model.named_parameters():
        if 'weight' in name and param.requires_grad:
            weight_std = param.data.std().item()
            expected_xavier = (2.0 / (param.shape[0] + param.shape[-1])) ** 0.5
            
            print(f"\n  {name}:")
            print(f"    Current std: {weight_std:.6f}")
            print(f"    Xavier std:  {expected_xavier:.6f}")
            
            if abs(weight_std - expected_xavier) < 0.1:
                print(f"    ✅ Likely Xavier initialization")
            elif weight_std < 0.01:
                print(f"    ⚠️  Very small weights (possible zero init)")
            elif weight_std > 1.0:
                print(f"    ⚠️  Very large weights (possible issue)")
            else:
                print(f"    ℹ️  Custom initialization")
    
    print(f"\n" + "=" * 80)
    print("DRY RUN COMPLETE")
    print("=" * 80)
    print("\n💡 Key Findings:")
    print(f"   - Model has {trainable_params:,} trainable parameters")
    print(f"   - Using {type(model.optimizer).__name__} optimizer")
    print(f"   - Learning rate: {model.optimizer.param_groups[0]['lr']}")
    print(f"   - Weight decay: {model.optimizer.param_groups[0].get('weight_decay', 'N/A')}")
    
    if os.path.exists(production_model_path):
        print(f"   - Pre-trained weights available and loaded")
    else:
        print(f"   - Using random weight initialization")
    
    print(f"\n🚀 To run actual training:")
    print(f"   python main.py -p")
    
    return model

if __name__ == "__main__":
    model = dry_run_process_step()