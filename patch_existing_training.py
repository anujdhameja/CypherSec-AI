"""
Quick Patch for Existing Training Code
This adds all critical fixes to your current training loop

Run this BEFORE running your hyperparameter search:
    python patch_existing_training.py

This will create a backup and patch your modeling.py file with:
1. Gradient clipping
2. Class weights  
3. Learning rate scheduling
4. Enhanced monitoring
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create backup of file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ Backup created: {backup_path}")
    return backup_path

def find_modeling_file():
    """Find the modeling.py file"""
    possible_paths = [
        'src/process/modeling.py',
        'src/process/model.py',
        'modeling.py',
        'model.py'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def add_gradient_clipping(content):
    """Add gradient clipping to training loop"""
    
    # Pattern to find loss.backward() followed by optimizer.step()
    pattern = r'(loss\.backward\(\))\s*\n\s*(optimizer\.step\(\))'
    
    replacement = r'''\1
            
            # PATCH: Gradient clipping to prevent explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=self.config.get('gradient_clip', 1.0)
            )
            
            # Log large gradients
            if grad_norm > 10.0:
                print(f"⚠️ Large gradient norm: {grad_norm:.2f} at batch {batch_idx}")
            
            \2'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✓ Added gradient clipping")
    else:
        print("⚠️  Could not find loss.backward() -> optimizer.step() pattern")
    
    return content

def add_class_weights(content):
    """Add class weights to loss function"""
    
    # Pattern to find CrossEntropyLoss initialization
    pattern = r'(criterion\s*=\s*nn\.CrossEntropyLoss\(\))'
    
    replacement = r'''# PATCH: Calculate class weights for imbalanced data
        try:
            import pandas as pd
            train_df = pd.read_csv('data/input/train.csv')
            class_counts = train_df['target'].value_counts()
            total = len(train_df)
            
            weight_0 = total / (2 * class_counts[0])
            weight_1 = total / (2 * class_counts[1])
            class_weights = torch.tensor([weight_0, weight_1]).to(self.device)
            
            print(f"Using class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        except Exception as e:
            print(f"Could not calculate class weights: {e}")
            \1'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✓ Added class weights")
    else:
        print("⚠️  Could not find CrossEntropyLoss initialization")
    
    return content

def add_scheduler(content):
    """Add learning rate scheduler"""
    
    # Pattern to find optimizer initialization
    pattern = r'(optimizer\s*=\s*[^\n]+\n)'
    
    if re.search(pattern, content):
        # Check if scheduler already exists
        if 'scheduler' not in content:
            addition = r'''\1
        
        # PATCH: Add learning rate scheduler
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        print("✓ Learning rate scheduler initialized")
        '''
            
            content = re.sub(pattern, addition, content, count=1)
            print("✓ Added LR scheduler initialization")
        
        # Add scheduler step in validation
        val_pattern = r'(val_loss\s*=\s*[^\n]+)'
        if re.search(val_pattern, content) and 'scheduler.step' not in content:
            scheduler_step = r'''\1
            
            # PATCH: Update learning rate based on validation loss
            if 'scheduler' in locals():
                scheduler.step(val_loss)'''
            
            content = re.sub(val_pattern, scheduler_step, content, count=1)
            print("✓ Added LR scheduler step")
    
    return content

def add_monitoring(content):
    """Add enhanced monitoring"""
    
    # Add unique prediction check
    pattern = r'(val_acc\s*=\s*[^\n]+)'
    
    monitoring_code = r'''\1
        
        # PATCH: Check for model collapse
        _, predicted = torch.max(outputs, 1)
        unique_preds = len(set(predicted.cpu().numpy()))
        
        if unique_preds == 1:
            print(f"❌ WARNING: Model predicting only class {predicted[0].item()}!")
        
        # Per-class accuracy
        for cls in [0, 1]:
            mask = (targets == cls)
            if mask.sum() > 0:
                cls_acc = ((predicted == targets) & mask).sum().item() / mask.sum().item()
                print(f"  Class {cls} accuracy: {cls_acc:.4f}")'''
    
    if re.search(pattern, content):
        content = re.sub(pattern, monitoring_code, content)
        print("✓ Added enhanced monitoring")
    
    return content

def add_imports(content):
    """Ensure necessary imports are present"""
    
    imports_to_add = []
    
    if 'import torch.nn as nn' not in content:
        imports_to_add.append('import torch.nn as nn')
    
    if 'from torch.optim.lr_scheduler import' not in content:
        imports_to_add.append('from torch.optim.lr_scheduler import ReduceLROnPlateau')
    
    if imports_to_add:
        # Add at the beginning after existing imports
        import_section = '\n'.join(imports_to_add) + '\n\n'
        
        # Find first import statement
        import_pattern = r'(^import\s+[^\n]+\n)'
        if re.search(import_pattern, content, re.MULTILINE):
            content = re.sub(import_pattern, r'\1' + import_section, content, count=1)
            print(f"✓ Added {len(imports_to_add)} missing imports")
    
    return content

def patch_file(filepath):
    """Apply all patches to file"""
    
    print(f"\nPatching {filepath}...")
    print("-" * 60)
    
    # Read original content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_path = backup_file(filepath)
    
    # Apply patches
    content = add_imports(content)
    content = add_gradient_clipping(content)
    content = add_class_weights(content)
    content = add_scheduler(content)
    content = add_monitoring(content)
    
    # Write patched content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("-" * 60)
    print(f"✓ File patched successfully!")
    print(f"✓ Original saved to: {backup_path}")
    
    return True

def create_gradient_clip_config():
    """Add gradient_clip to configs.json if missing"""
    
    config_path = 'configs.json'
    
    if not os.path.exists(config_path):
        print("⚠️  configs.json not found")
        return
    
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add gradient_clip if missing
    if 'process' in config:
        if 'gradient_clip' not in config['process']:
            config['process']['gradient_clip'] = 1.0
            
            # Backup and save
            backup_file(config_path)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✓ Added gradient_clip to configs.json")

def verify_patches():
    """Verify that patches were applied correctly"""
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    modeling_file = find_modeling_file()
    
    if not modeling_file:
        print("❌ Could not find modeling file")
        return False
    
    with open(modeling_file, 'r') as f:
        content = f.read()
    
    checks = {
        'Gradient Clipping': 'clip_grad_norm_' in content,
        'Class Weights': 'class_weights' in content or 'weight_0' in content,
        'LR Scheduler': 'ReduceLROnPlateau' in content or 'scheduler' in content,
        'Model Collapse Check': 'unique_preds' in content or 'unique predictions' in content.lower(),
    }
    
    all_passed = True
    
    for check_name, passed in checks.items():
        status = "✓" if passed else "❌"
        print(f"{status} {check_name}: {'Present' if passed else 'Missing'}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All patches verified successfully!")
        print("\nYou can now run:")
        print("  python main.py -p")
        print("  or")
        print("  python auto_hyperparameter_comprehensive.py")
    else:
        print("\n⚠️  Some patches may not have been applied correctly.")
        print("You may need to manually add missing fixes.")
    
    return all_passed

def main():
    """Main execution"""
    
    print("="*60)
    print("devign Training Code Patcher")
    print("="*60)
    print("\nThis will patch your training code with critical fixes:")
    print("  1. Gradient clipping")
    print("  2. Class weights")
    print("  3. Learning rate scheduling")
    print("  4. Enhanced monitoring")
    print("\nBackups will be created automatically.")
    print("="*60 + "\n")
    
    # Find modeling file
    modeling_file = find_modeling_file()
    
    if not modeling_file:
        print("❌ Could not find modeling.py file")
        print("\nSearched in:")
        print("  - src/process/modeling.py")
        print("  - src/process/model.py")
        print("  - modeling.py")
        print("  - model.py")
        print("\nPlease run this script from the project root directory.")
        return
    
    print(f"Found modeling file: {modeling_file}")
    
    # Confirm
    response = input("\nContinue with patching? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Patch file
    try:
        patch_file(modeling_file)
        
        # Update configs
        create_gradient_clip_config()
        
        # Verify
        verify_patches()
        
    except Exception as e:
        print(f"\n❌ Error during patching: {e}")
        import traceback
        traceback.print_exc()
        print("\nYou may need to apply patches manually.")
        print("See the backup file for the original code.")

if __name__ == "__main__":
    main()