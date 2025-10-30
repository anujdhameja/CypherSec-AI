#!/usr/bin/env python3
"""
Show Model Location and Configuration
"""

import os
import configs
from pathlib import Path

def show_model_info():
    """Display model location and configuration details"""
    
    print("="*80)
    print("MODEL LOCATION AND CONFIGURATION")
    print("="*80)
    
    # Load configurations
    PATHS = configs.Paths()
    FILES = configs.Files()
    
    # Model path information
    model_dir = PATHS.model
    model_filename = FILES.model
    model_path = model_dir + model_filename
    
    print(f"\n📁 MODEL DIRECTORY:")
    print(f"   Relative path: {model_dir}")
    print(f"   Absolute path: {os.path.abspath(model_dir)}")
    
    print(f"\n📄 MODEL FILE:")
    print(f"   Filename: {model_filename}")
    print(f"   Full relative path: {model_path}")
    print(f"   Full absolute path: {os.path.abspath(model_path)}")
    
    # Check if directory exists
    if os.path.exists(model_dir):
        print(f"   ✅ Model directory exists")
    else:
        print(f"   ⚠️  Model directory does not exist - will be created during training")
        os.makedirs(model_dir, exist_ok=True)
        print(f"   ✅ Created model directory: {os.path.abspath(model_dir)}")
    
    # Check if model file exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"   ✅ Model file exists ({file_size:,} bytes)")
    else:
        print(f"   ℹ️  Model file does not exist yet - will be created after training")
    
    # List all files in model directory
    print(f"\n📋 FILES IN MODEL DIRECTORY:")
    if os.path.exists(model_dir):
        files = list(Path(model_dir).glob("*"))
        if files:
            for file in sorted(files):
                size = file.stat().st_size if file.is_file() else 0
                file_type = "📄" if file.is_file() else "📁"
                print(f"   {file_type} {file.name} ({size:,} bytes)")
        else:
            print(f"   (Directory is empty)")
    
    # Training configuration
    context = configs.Process()
    devign_config = configs.Devign()
    
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print(f"   Learning Rate: {devign_config.learning_rate}")
    print(f"   Weight Decay: {devign_config.weight_decay}")
    print(f"   Batch Size: {context.batch_size}")
    print(f"   Max Epochs: {context.epochs}")
    print(f"   Patience: {context.patience}")
    
    print(f"\n🚀 TRAINING COMMANDS:")
    print(f"   Start training: python main.py -p")
    print(f"   Evaluate model: python evaluate_model_comprehensive.py")
    
    print(f"\n💾 MODEL SAVING DETAILS:")
    print(f"   During training: Model will be saved to {os.path.abspath(model_path)}")
    print(f"   After training: Final model available at {os.path.abspath(model_path)}")
    print(f"   Backup models: Check {os.path.abspath(model_dir)} for additional saved models")
    
    return model_path, os.path.abspath(model_path)

if __name__ == "__main__":
    relative_path, absolute_path = show_model_info()
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Final model will be saved as: final_model.pth")
    print(f"📍 Location: {absolute_path}")
    print(f"🔧 Ready for training with: python main.py -p")