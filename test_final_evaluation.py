#!/usr/bin/env python3
"""
Test Final Evaluation on Saved Model
Run comprehensive evaluation without retraining
"""

import torch
import sys
import os

# Add src to path
sys.path.append('src')

import configs
from src.process.devign import Devign
from src.process.modeling import predict
from src.process.loader_step import LoaderStep
import src.data as data

def test_saved_model_evaluation():
    """Test the comprehensive evaluation on the saved model"""
    
    print("="*80)
    print("TESTING FINAL EVALUATION ON SAVED MODEL")
    print("="*80)
    
    # Load configurations
    context = configs.Process()
    devign_config = configs.Devign()
    PATHS = configs.Paths()
    FILES = configs.Files()
    DEVICE = FILES.get_device()
    
    # Load the saved model
    model_path = PATHS.model + FILES.model
    print(f"Loading model from: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Create model instance
    model = Devign(
        path=model_path, 
        device=DEVICE, 
        model=devign_config.model, 
        learning_rate=devign_config.learning_rate,
        weight_decay=devign_config.weight_decay,
        loss_lambda=devign_config.loss_lambda
    )
    
    # Load the trained weights
    try:
        model.load()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data (same as training)
    print(f"\n📊 Loading test dataset...")
    input_dataset = data.loads(PATHS.input)
    
    # Split the dataset (same as training)
    train_loader, val_loader, test_loader = list(
        map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
            data.train_val_test_split(input_dataset, shuffle=context.shuffle)))
    
    test_loader_step = LoaderStep("Test", test_loader, DEVICE)
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Run comprehensive evaluation
    print(f"\n🧪 Running comprehensive evaluation...")
    
    try:
        test_accuracy = predict(model, test_loader_step)
        print(f"\n🎯 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        if test_accuracy > 0.80:
            print(f"🎉 EXCELLENT: Model achieved target performance!")
        elif test_accuracy > 0.75:
            print(f"✅ VERY GOOD: Model performance is strong!")
        else:
            print(f"👍 GOOD: Model performance is solid!")
            
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_saved_model_evaluation()