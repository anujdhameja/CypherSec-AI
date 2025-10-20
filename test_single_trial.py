"""
Quick test to see if a single trial works now
This will run just 1 trial to verify the fix worked
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*70)
print("TESTING SINGLE TRIAL")
print("="*70)
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    # Import your modules
    from src.utils.objects.input_dataset import InputDataset
    from src.process.devign import Devign
    
    print("✓ Imports successful")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load small sample of data
    print("\n1. Loading data...")
    data_path = os.path.join('data', 'input')
    print(f"Loading data from: {os.path.abspath(data_path)}")
    
    # Load only first 2 files for testing (to make it faster)
    dataset = InputDataset(data_path, max_files=2)
    
    # Get a data loader
    batch_size = 8
    train_loader = dataset.get_loader(batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Print dataset info
    print(f"\nDataset info:")
    print(f"- Total samples: {len(dataset)}")
    print(f"- Number of batches: {len(train_loader)}")
    print(f"- Batch size: {batch_size}")
    
    # Check first batch
    print("\nChecking first batch...")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys}")
        print(f"Batch size: {batch.num_graphs}")
        print(f"Features shape: {batch.x.shape if hasattr(batch, 'x') else 'N/A'}")
        print(f"Edge index shape: {batch.edge_index.shape if hasattr(batch, 'edge_index') else 'N/A'}")
        print(f"Targets: {batch.y.tolist() if hasattr(batch, 'y') else 'N/A'}")
        break  # Just check first batch
    
    # Create model configuration
    print("\n2. Creating model...")
    
    model_config = {
        'conv_args': {
            'conv1d_1': {'in_channels': 100, 'out_channels': 100, 'kernel_size': 3},
        },
        'gated_graph_conv_args': {
            'out_channels': 200,
            'num_layers': 4
        }
    }
    
    # Initialize the model
    model = Devign(
        path='./saved_models',
        device=device,
        model=model_config,
        learning_rate=1e-4,
        weight_decay=1e-6,
        loss_lambda=0.5
    )
    model = model.model  # Get the underlying PyTorch model
    model.to(device)
    
    print(f"✓ Model created on {device}")
    
    # Setup training
    print("\n3. Setting up training...")
    
    # Calculate class weights
    all_targets = []
    for batch in train_loader:
        all_targets.extend(batch.y.cpu().numpy())
    
    import numpy as np
    unique, counts = np.unique(all_targets, return_counts=True)
    total = len(all_targets)
    weights = torch.zeros(2)
    for cls, count in zip(unique, counts):
        weights[int(cls)] = total / (2 * count)
    
    class_weights = weights.to(device)
    print(f"✓ Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Try one batch
    print("\n4. Testing single batch...")
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Get targets
        targets = batch.y
        print(f"   Target dtype before: {targets.dtype}")
        
        # Convert to long (THIS IS THE FIX)
        targets = targets.long()
        print(f"   Target dtype after: {targets.dtype}")
        print(f"   Target values: {targets[:5]}")
        
        # Compute loss (THIS IS WHERE IT WAS FAILING)
        try:
            loss = criterion(outputs, targets)
            print(f"   ✓ Loss computed: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            print(f"   ✓ Gradient norm: {grad_norm:.2f}")
            
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).sum().item() / targets.size(0)
            print(f"   ✓ Accuracy: {accuracy:.4f}")
            
            print("\n✅ SUCCESS! Single batch works perfectly!")
            print("\nThe fix is working. Your hyperparameter search should now run.")
            break
            
        except RuntimeError as e:
            print(f"\n❌ FAILED: {e}")
            print("\nThe fix didn't work. Need to debug further.")
            sys.exit(1)
    
    # Try full epoch
    print("\n5. Testing full epoch...")
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in train_loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch)
        targets = batch.y.long()  # THE FIX
        
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    unique_preds = len(set(all_preds))
    
    print(f"✓ Full epoch completed!")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Unique predictions: {unique_preds}")
    
    if unique_preds == 1:
        print("  ⚠️  Warning: Only predicting 1 class")
    else:
        print("  ✓ Predicting both classes")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nYour auto_hyperparameter_comprehensive.py should now work!")
    print("Run: python auto_hyperparameter_comprehensive.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nDebugging info:")
    print("Make sure you're running from the project root directory")