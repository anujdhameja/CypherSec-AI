
# """
# Run This Diagnostic Script
# Integrates with your existing training code
# """

# import torch
# from torch_geometric.loader import DataLoader
# import pandas as pd

# # Import your model
# from src.process.model import DevignModel  # Adjust import as needed
# from balanced_training_config import BalancedDevignModel

# # Import diagnostic functions
# from critical_diagnostic import (
#     run_full_diagnostic,
#     single_batch_overfit_test
# )

# def main():
#     print("\n" + "#"*80)
#     print("# DIAGNOSTIC RUN")
#     print("#"*80)
    
#     # Load data
#     print("\nLoading data...")
    
#     # Load your processed input files
#     import sys
#     sys.path.append('.')
#     from src import data
#     from paths import PATHS
    
#     input_dataset = data.loads(PATHS.input)
    
#     # Split data (same as your training)
#     train_dataset, val_dataset, test_dataset = data.train_val_test_split(
#         input_dataset, shuffle=False
#     )
    
#     # Create loaders
#     train_loader = train_dataset.get_loader(batch_size=8, shuffle=False)
#     val_loader = val_dataset.get_loader(batch_size=8, shuffle=False)
    
#     print(f"✓ Train samples: {len(train_dataset)}")
#     print(f"✓ Val samples: {len(val_dataset)}")
    
#     # Create model
#     print("\nCreating model...")
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     model = BalancedDevignModel(
#         input_dim=205,
#         output_dim=2,
#         hidden_dim=200,
#         num_steps=4,
#         dropout=0.4
#     ).to(device)
    
#     # Try to load trained weights if they exist
#     try:
#         model.load('data/model/devign.model')
#         print("✓ Loaded trained model")
#     except:
#         print("⚠️ Using random initialization (no trained model found)")
    
#     # Run diagnostics
#     print("\n" + "="*80)
#     print("RUNNING DIAGNOSTICS")
#     print("="*80)
    
#     # Full diagnostic
#     results = run_full_diagnostic(model, train_loader, val_loader, device)
    
#     # Single batch test
#     print("\n" + "="*80)
#     single_batch_overfit_test(model, train_loader, device, num_steps=100)
    
#     print("\n" + "#"*80)
#     print("# DIAGNOSTIC COMPLETE")
#     print("#"*80)
#     print("\nCheck the output above for issues and recommendations.")


# if __name__ == "__main__":
#     main()



"""
Run This Diagnostic Script
Integrates with your existing training code
"""

import os
import sys
import torch
from torch_geometric.loader import DataLoader
import pandas as pd

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(script_dir))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import your model
try:
    from src.process.model import DevignModel
    from src.process.balanced_training_config import BalancedDevignModel
    from src.process.critical_diagonostic import (
        run_full_diagnostic,
        single_batch_overfit_test
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Current Python path:", sys.path)
    raise

def main():
    print("\n" + "#"*80)
    print("# DIAGNOSTIC RUN")
    print("#"*80)
    
    # Load data
    print("\nLoading data...")
    
    # Load your processed input files
    from src import data
    from configs import Paths
    PATHS = Paths()
    
    input_dataset = data.loads(PATHS.input)
    
    # Split data (same as your training)
    train_dataset, val_dataset, test_dataset = data.train_val_test_split(
        input_dataset, shuffle=False
    )
    
    # Create loaders
    train_loader = train_dataset.get_loader(batch_size=8, shuffle=False)
    val_loader = val_dataset.get_loader(batch_size=8, shuffle=False)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BalancedDevignModel(
        input_dim=205,
        output_dim=2,
        hidden_dim=200,
        num_steps=4,
        dropout=0.4
    ).to(device)
    
    # Try to load trained weights if they exist
    try:
        model.load('data/model/devign.model')
        print("✓ Loaded trained model")
    except:
        print("⚠️ Using random initialization (no trained model found)")
    
    # Run diagnostics
    print("\n" + "="*80)
    print("RUNNING DIAGNOSTICS")
    print("="*80)
    
    # Full diagnostic
    results = run_full_diagnostic(model, train_loader, val_loader, device)
    
    # Single batch test
    print("\n" + "="*80)
    single_batch_overfit_test(model, train_loader, device, num_steps=100)
    
    print("\n" + "#"*80)
    print("# DIAGNOSTIC COMPLETE")
    print("#"*80)
    print("\nCheck the output above for issues and recommendations.")

if __name__ == "__main__":
    main()