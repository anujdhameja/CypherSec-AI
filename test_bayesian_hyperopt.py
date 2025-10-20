#!/usr/bin/env python3
"""
Test script for the updated Bayesian Hyperparameter Optimization
"""

import sys
import torch
from pathlib import Path

def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        # Test core imports
        import configs
        import src.data as data
        import src.process as process
        from src.process.balanced_training_config import BalancedDevignModel
        from src.process.step import Step
        from src.process.modeling import Train
        
        print("✓ Core imports successful")
        
        # Test Bayesian optimization imports
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
        from skopt.callbacks import CheckpointSaver
        
        print("✓ Bayesian optimization imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test that the model can be created with the new architecture"""
    print("\nTesting model creation...")
    
    try:
        from src.process.balanced_training_config import BalancedDevignModel
        import configs
        
        # Get embedding config
        embed_config = configs.Embed()
        
        # Create model
        model = BalancedDevignModel(
            input_dim=embed_config.nodes_dim,  # 100
            output_dim=2,
            hidden_dim=200,
            num_steps=4,
            dropout=0.3
        )
        
        print(f"✓ Model created successfully")
        print(f"   Input dim: {embed_config.nodes_dim}")
        print(f"   Hidden dim: 200")
        print(f"   Output dim: 2")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_data_loading():
    """Test that data can be loaded correctly"""
    print("\nTesting data loading...")
    
    try:
        import configs
        import src.data as data
        
        # Get paths
        PATHS = configs.Paths()
        
        # Check if input data exists
        input_path = Path(PATHS.input)
        if not input_path.exists():
            print(f"⚠️ Input data directory not found: {input_path}")
            print("   This is expected if you haven't run the embedding step yet")
            return True
        
        # Try to load data
        input_dataset = data.loads(PATHS.input)
        print(f"✓ Data loading successful")
        print(f"   Found {len(input_dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_bayesian_class():
    """Test that the BayesianHyperparameterSearch class can be instantiated"""
    print("\nTesting BayesianHyperparameterSearch class...")
    
    try:
        # Import the updated class
        sys.path.append('.')
        from auto_hyperparameter_bayesian import BayesianHyperparameterSearch
        
        # Try to create instance (but don't load data)
        print("✓ BayesianHyperparameterSearch class imported successfully")
        
        # Test search space definition
        search = BayesianHyperparameterSearch.__new__(BayesianHyperparameterSearch)
        search.results = {'search_space': None}
        search_space = search.define_search_space()
        
        print(f"✓ Search space defined with {len(search_space)} dimensions:")
        for dim in search_space:
            print(f"   - {dim.name}: {dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ BayesianHyperparameterSearch error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("TESTING UPDATED BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_loading,
        test_bayesian_class
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! The updated Bayesian optimization should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    print("\nTo run the Bayesian optimization:")
    print("python auto_hyperparameter_bayesian.py")

if __name__ == "__main__":
    main()