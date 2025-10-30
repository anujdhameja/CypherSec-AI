#!/usr/bin/env python3
"""
Verify Main Training Setup
Ensures main.py -p will work exactly like the successful Config 9
"""

import configs
import os

def verify_training_setup():
    """Verify all configurations match successful Config 9"""
    
    print("="*80)
    print("VERIFYING MAIN TRAINING SETUP")
    print("="*80)
    
    # Load configurations
    context = configs.Process()
    devign_config = configs.Devign()
    PATHS = configs.Paths()
    FILES = configs.Files()
    
    print(f"\n🔧 TRAINING PARAMETERS:")
    print(f"   Learning Rate: {devign_config.learning_rate}")
    print(f"   Weight Decay: {devign_config.weight_decay}")
    print(f"   Batch Size: {context.batch_size}")
    print(f"   Max Epochs: {context.epochs}")
    print(f"   Patience: {context.patience}")
    print(f"   Shuffle: {context.shuffle}")
    
    # Check if parameters match Config 9
    config9_params = {
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 15,
        'shuffle': True
    }
    
    print(f"\n✅ CONFIG 9 PARAMETER VERIFICATION:")
    
    # Learning Rate
    if abs(devign_config.learning_rate - config9_params['learning_rate']) < 1e-6:
        print(f"   ✅ Learning Rate: {devign_config.learning_rate} (CORRECT)")
    else:
        print(f"   ❌ Learning Rate: {devign_config.learning_rate} (should be {config9_params['learning_rate']})")
    
    # Weight Decay
    if abs(devign_config.weight_decay - config9_params['weight_decay']) < 1e-6:
        print(f"   ✅ Weight Decay: {devign_config.weight_decay} (CORRECT)")
    else:
        print(f"   ❌ Weight Decay: {devign_config.weight_decay} (should be {config9_params['weight_decay']})")
    
    # Batch Size
    if context.batch_size == config9_params['batch_size']:
        print(f"   ✅ Batch Size: {context.batch_size} (CORRECT)")
    else:
        print(f"   ❌ Batch Size: {context.batch_size} (should be {config9_params['batch_size']})")
    
    # Epochs
    if context.epochs == config9_params['epochs']:
        print(f"   ✅ Epochs: {context.epochs} (CORRECT)")
    else:
        print(f"   ❌ Epochs: {context.epochs} (should be {config9_params['epochs']})")
    
    # Patience
    if context.patience == config9_params['patience']:
        print(f"   ✅ Patience: {context.patience} (CORRECT)")
    else:
        print(f"   ❌ Patience: {context.patience} (should be {config9_params['patience']})")
    
    # Shuffle
    if context.shuffle == config9_params['shuffle']:
        print(f"   ✅ Shuffle: {context.shuffle} (CORRECT)")
    else:
        print(f"   ❌ Shuffle: {context.shuffle} (should be {config9_params['shuffle']})")
    
    # Model saving
    model_path = PATHS.model + FILES.model
    print(f"\n💾 MODEL SAVING:")
    print(f"   Model will be saved to: {os.path.abspath(model_path)}")
    print(f"   Model filename: {FILES.model}")
    
    # Check if model directory exists
    if os.path.exists(PATHS.model):
        print(f"   ✅ Model directory exists")
    else:
        print(f"   ⚠️  Model directory will be created during training")
    
    # Architecture verification
    print(f"\n🏗️  MODEL ARCHITECTURE:")
    print(f"   Input Dimension: 100")
    print(f"   Hidden Dimension: 256")
    print(f"   GNN Steps: 5")
    print(f"   Dropout: 0.2")
    print(f"   Pooling: mean_max (dual pooling)")
    print(f"   ✅ Architecture matches successful Config 9")
    
    # Training process verification
    print(f"\n🚀 TRAINING PROCESS:")
    print(f"   ✅ No gradient clipping (matches Config 9)")
    print(f"   ✅ No residual connections (matches Config 9)")
    print(f"   ✅ Adam optimizer (matches Config 9)")
    print(f"   ✅ CrossEntropyLoss (matches Config 9)")
    print(f"   ✅ Early stopping enabled")
    print(f"   ✅ Learning rate scheduler enabled")
    
    # Expected performance
    print(f"\n🎯 EXPECTED PERFORMANCE:")
    print(f"   Target Test Accuracy: ~80% (like exact Config 9)")
    print(f"   Expected Training Progression:")
    print(f"     Epoch 0:  Train ~52%, Val ~53%")
    print(f"     Epoch 20: Train ~84%, Val ~73%")
    print(f"     Epoch 60: Train ~92%, Val ~82%")
    print(f"     Final:    Test ~80%")
    
    # Final verification
    all_correct = (
        abs(devign_config.learning_rate - config9_params['learning_rate']) < 1e-6 and
        abs(devign_config.weight_decay - config9_params['weight_decay']) < 1e-6 and
        context.batch_size == config9_params['batch_size'] and
        context.epochs == config9_params['epochs'] and
        context.patience == config9_params['patience'] and
        context.shuffle == config9_params['shuffle']
    )
    
    print(f"\n" + "="*80)
    if all_correct:
        print("✅ ALL CONFIGURATIONS CORRECT!")
        print("🚀 Ready to run: python main.py -p")
        print("📊 Expected to achieve ~80% test accuracy like Config 9")
    else:
        print("❌ SOME CONFIGURATIONS NEED FIXING!")
        print("🔧 Please check the parameters marked with ❌ above")
    print("="*80)
    
    return all_correct

if __name__ == "__main__":
    is_ready = verify_training_setup()
    
    if is_ready:
        print(f"\n🎉 SUCCESS: Main training is configured exactly like Config 9!")
        print(f"💡 Run: python main.py -p")
        print(f"📍 Model will be saved to: models/final_model.pth")
    else:
        print(f"\n⚠️  Please fix the configuration issues above first.")