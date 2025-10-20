"""
Adapter Bridge for Devign Project
This file helps integrate the hyperparameter search with your existing code
Place this in your project root alongside auto_hyperparameter_comprehensive.py
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def get_data_loaders(batch_size=8):
    """
    Load data using your existing Devign project structure
    """
    try:
        # Import your actual data loading modules
        import src.data as data
        from src.utils.objects.input_dataset import InputDataset
        from src.utils.objects.paths import PATHS
        
        print("Loading data using Devign project structure...")
        
        # Load the input dataset (same as main.py)
        input_dataset = data.loads(PATHS.input)
        print(f"✓ Loaded input dataset: {len(input_dataset)} samples")
        
        # Split the dataset (same as main.py)
        train_dataset, test_dataset, val_dataset = data.train_val_test_split(
            input_dataset, shuffle=False
        )
        
        print(f"✓ Split data - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create data loaders (same as main.py)
        train_loader = train_dataset.get_loader(batch_size, shuffle=False)
        val_loader = val_dataset.get_loader(batch_size, shuffle=False)
        test_loader = test_dataset.get_loader(batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying alternative data loading...")
        
        # Fallback: Load from individual pickle files
        try:
            import pandas as pd
            import os
            from src.utils.objects.input_dataset import InputDataset
            
            # Load all input pickle files (like data_diagnostic.py)
            input_dir = "data/input"
            all_data = []
            
            pkl_files = [f for f in os.listdir(input_dir) if f.endswith("_cpg_input.pkl")]
            pkl_files.sort(key=lambda x: int(x.split('_')[0]))
            
            for pkl_file in pkl_files[:5]:  # Load first 5 files for testing
                file_path = os.path.join(input_dir, pkl_file)
                df = pd.read_pickle(file_path)
                all_data.append(df)
            
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✓ Loaded {len(combined_df)} samples from {len(all_data)} files")
            
            # Simple split
            from sklearn.model_selection import train_test_split
            
            train_data, temp_data = train_test_split(
                combined_df, test_size=0.3, stratify=combined_df['target'], random_state=42
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, stratify=temp_data['target'], random_state=42
            )
            
            # Create datasets
            train_dataset = InputDataset(train_data)
            val_dataset = InputDataset(val_data)
            test_dataset = InputDataset(test_data)
            
            # Create loaders
            train_loader = train_dataset.get_loader(batch_size, shuffle=True)
            val_loader = val_dataset.get_loader(batch_size, shuffle=False)
            test_loader = test_dataset.get_loader(batch_size, shuffle=False)
            
            return train_loader, val_loader, test_loader
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise NotImplementedError("Could not load data with any method")

def create_model(config):
    """
    Create model using your existing Devign project structure
    Returns the raw PyTorch model for hyperparameter search
    """
    try:
        # Create the raw model directly (not wrapped in Devign class)
        from src.process.balanced_training_config import BalancedDevignModel
        import torch
        
        model = BalancedDevignModel(
            input_dim=100,  # Based on data diagnostic
            output_dim=2,
            hidden_dim=config.get('hidden_dim', 200),
            num_steps=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.3)
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        print(f"✓ Created BalancedDevignModel with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        return model
        
    except ImportError as e:
        print(f"Error importing BalancedDevignModel: {e}")
        
        # Fallback: Create simple model using stable_training
        try:
            from stable_training import StableDevignModel
            import torch
            
            model = StableDevignModel(
                input_dim=100,  # Based on data diagnostic
                output_dim=2,
                hidden_dim=config.get('hidden_dim', 200),
                num_steps=config.get('num_layers', 4),
                dropout=config.get('dropout', 0.3)
            )
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            
            print(f"✓ Created StableDevignModel with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            return model
            
        except ImportError as e2:
            print(f"All model creation methods failed: {e2}")
            raise

def _get_input_dim_from_loader(train_loader):
    """Extract input dimension from data loader"""
    for batch in train_loader:
        if hasattr(batch, 'x'):
            return batch.x.size(1)
        elif hasattr(batch, 'features'):
            return batch.features.size(1)
        elif isinstance(batch, tuple):
            return batch[0].size(1)
        break
    
    # Based on data diagnostic - your data has 100-dim features
    return 100

def get_class_weights(train_loader, device):
    """Calculate class weights from training data"""
    import torch
    import numpy as np
    
    all_labels = []
    
    for batch in train_loader:
        # Adjust this based on how your data is structured
        if hasattr(batch, 'y'):
            all_labels.extend(batch.y.cpu().numpy())
        elif hasattr(batch, 'label'):
            all_labels.extend(batch.label.cpu().numpy())
        elif isinstance(batch, tuple):
            all_labels.extend(batch[1].cpu().numpy())
    
    # Calculate weights
    all_labels = np.array(all_labels)
    unique, counts = np.unique(all_labels, return_counts=True)
    
    total = len(all_labels)
    weights = torch.zeros(2)
    
    for cls, count in zip(unique, counts):
        weights[int(cls)] = total / (2 * count)
    
    print(f"Calculated class weights: {weights}")
    return weights.to(device)

def extract_targets_from_batch(batch):
    """Extract target labels from batch - Devign project specific"""
    # In your project, batch is a PyTorch Geometric Data object
    if hasattr(batch, 'y'):
        return batch.y
    elif hasattr(batch, 'target'):
        return batch.target
    elif isinstance(batch, tuple):
        return batch[1]
    else:
        raise ValueError(f"Cannot extract targets from batch of type {type(batch)}")

# Configuration template
DEFAULT_CONFIG = {
    'learning_rate': 1e-4,
    'gradient_clip': 1.0,
    'dropout': 0.3,
    'weight_decay': 1.3e-6,
    'hidden_dim': 128,
    'num_layers': 3,
    'batch_size': 8,
    'optimizer': 'adam',
    'class_weight_scale': 1.0,
    'scheduler': 'plateau',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5
}

if __name__ == "__main__":
    # Test the adapter
    print("Testing adapter...")
    
    try:
        print("\n1. Testing data loading...")
        train_loader, val_loader, test_loader = get_data_loaders()
        print(f"   ✓ Train batches: {len(train_loader)}")
        print(f"   ✓ Val batches: {len(val_loader)}")
        print(f"   ✓ Test batches: {len(test_loader)}")
        
        print("\n2. Testing model creation...")
        model = create_model(DEFAULT_CONFIG)
        print(f"   ✓ Model created: {type(model)}")
        
        print("\n3. Testing class weights...")
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = get_class_weights(train_loader, device)
        print(f"   ✓ Class weights: {weights}")
        
        print("\n4. Testing batch structure...")
        for batch in train_loader:
            targets = extract_targets_from_batch(batch)
            print(f"   ✓ Batch shape: {batch.x.shape if hasattr(batch, 'x') else 'Unknown'}")
            print(f"   ✓ Target shape: {targets.shape}")
            break
        
        print("\n✓ All tests passed! Adapter is ready.")
        print("\nYou can now run:")
        print("  python auto_hyperparameter_comprehensive.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nPlease modify adapter.py to match your code structure.")
        import traceback
        traceback.print_exc()