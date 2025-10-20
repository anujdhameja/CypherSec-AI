# """
# HOTFIX: Fix the "expected scalar type Long but found Float" error
# This patches auto_hyperparameter_comprehensive.py to convert targets to Long type

# Run: python hotfix_target_dtype.py
# """

# import os
# import shutil
# from datetime import datetime

# def apply_hotfix():
#     """Apply hotfix to auto_hyperparameter_comprehensive.py"""
    
#     filepath = 'auto_hyperparameter_comprehensive.py'
    
#     if not os.path.exists(filepath):
#         print(f"❌ File not found: {filepath}")
#         return False
    
#     print("="*60)
#     print("APPLYING HOTFIX FOR TARGET DTYPE ERROR")
#     print("="*60)
    
#     # Backup
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     backup_path = f"{filepath}.backup_{timestamp}"
#     shutil.copy2(filepath, backup_path)
#     print(f"✓ Backup created: {backup_path}")
    
#     # Read file
#     with open(filepath, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     # Fix 1: In train_epoch method
#     old_pattern_1 = '''            # Get targets
#             if hasattr(batch, 'y'):
#                 targets = batch.y
#             else:
#                 targets = batch[1]
            
#             # Compute loss
#             loss = criterion(outputs, targets)'''
    
#     new_pattern_1 = '''            # Get targets
#             if hasattr(batch, 'y'):
#                 targets = batch.y
#             else:
#                 targets = batch[1]
            
#             # HOTFIX: Convert targets to Long type for CrossEntropyLoss
#             targets = targets.long()
            
#             # Compute loss
#             loss = criterion(outputs, targets)'''
    
#     if old_pattern_1 in content:
#         content = content.replace(old_pattern_1, new_pattern_1, 1)
#         print("✓ Fixed train_epoch method")
#     else:
#         print("⚠️  Could not find train_epoch pattern (may already be fixed)")
    
#     # Fix 2: In validate_epoch method
#     old_pattern_2 = '''                if hasattr(batch, 'y'):
#                     targets = batch.y
#                 else:
#                     targets = batch[1]
                
#                 loss = criterion(outputs, targets)'''
    
#     new_pattern_2 = '''                if hasattr(batch, 'y'):
#                     targets = batch.y
#                 else:
#                     targets = batch[1]
                
#                 # HOTFIX: Convert targets to Long type for CrossEntropyLoss
#                 targets = targets.long()
                
#                 loss = criterion(outputs, targets)'''
    
#     if old_pattern_2 in content:
#         content = content.replace(old_pattern_2, new_pattern_2, 1)
#         print("✓ Fixed validate_epoch method")
#     else:
#         print("⚠️  Could not find validate_epoch pattern (may already be fixed)")
    
#     # Write back
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write(content)
    
#     print("="*60)
#     print("✓ HOTFIX APPLIED SUCCESSFULLY")
#     print("="*60)
#     print("\nYou can now run:")
#     print("  python auto_hyperparameter_comprehensive.py")
#     print("\nThe error 'expected scalar type Long but found Float' should be fixed.")
#     print("="*60)
    
#     return True

# if __name__ == "__main__":
#     apply_hotfix()



#version 2



"""
Complete Hotfix v2 - Finds and fixes ALL target conversion issues
Run: python hotfix_v2_complete.py
"""

import re
import os
import shutil
from datetime import datetime

def apply_complete_hotfix():
    """Apply comprehensive hotfix using regex patterns"""
    
    filepath = 'auto_hyperparameter_comprehensive.py'
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    print("="*70)
    print("APPLYING COMPLETE HOTFIX V2")
    print("="*70)
    
    # Backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✓ Backup created: {backup_path}")
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixes_applied = 0
    
    # Pattern 1: Find any "loss = criterion(outputs, targets)" without .long() before it
    # This uses regex to find and fix
    pattern1 = r'(targets\s*=\s*(?:batch\.y|batch\[1\]))\s*\n\s*\n\s*(# Compute loss|# Backward pass)?\s*\n?\s*(loss\s*=\s*criterion\(outputs,\s*targets\))'
    
    def replacement1(match):
        nonlocal fixes_applied
        fixes_applied += 1
        return f'''{match.group(1)}
            
            # HOTFIX: Convert targets to Long type for CrossEntropyLoss
            targets = targets.long()
            
            {match.group(2) if match.group(2) else ''}
            {match.group(3)}'''
    
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: More aggressive - any "loss = criterion(" without .long() nearby
    pattern2 = r'(targets\s*=\s*(?:batch\.y|batch\[1\]))\s*\n\s*((?:.*\n){0,3}?)\s*(loss\s*=\s*criterion\(outputs,\s*targets\))'
    
    def check_and_fix(match):
        nonlocal fixes_applied
        full_text = match.group(0)
        # Only fix if .long() is not already present
        if 'targets.long()' not in full_text and '.long()' not in full_text:
            fixes_applied += 1
            return f'''{match.group(1)}
            
            # HOTFIX: Convert targets to Long type for CrossEntropyLoss
            targets = targets.long()
            
            {match.group(2)}{match.group(3)}'''
        return full_text
    
    content = re.sub(pattern2, check_and_fix, content)
    
    # Check if any changes were made
    if content == original_content:
        print("⚠️  No patterns found to fix!")
        print("Checking if already fixed...")
        
        # Count existing .long() conversions
        long_count = content.count('targets.long()')
        print(f"Found {long_count} existing .long() conversions")
        
        if long_count >= 2:
            print("✓ File appears to already be fixed!")
            return True
        else:
            print("\n❌ Manual fix needed!")
            print("\nSearch for these lines in the file:")
            print("  1. In train_epoch method:")
            print("     targets = batch.y")
            print("     loss = criterion(outputs, targets)")
            print("\n  2. In validate_epoch method:")
            print("     targets = batch.y")
            print("     loss = criterion(outputs, targets)")
            print("\nAdd this line between them:")
            print("     targets = targets.long()")
            return False
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Applied {fixes_applied} fixes")
    
    # Verify
    with open(filepath, 'r', encoding='utf-8') as f:
        verify_content = f.read()
    
    long_count = verify_content.count('targets.long()')
    print(f"✓ Verification: Found {long_count} .long() conversions")
    
    print("="*70)
    print("✓ COMPLETE HOTFIX APPLIED SUCCESSFULLY")
    print("="*70)
    
    return True

def show_affected_lines():
    """Show the affected lines in the file"""
    filepath = 'auto_hyperparameter_comprehensive.py'
    
    if not os.path.exists(filepath):
        return
    
    print("\nShowing affected areas:")
    print("-"*70)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        if 'targets.long()' in line:
            print(f"\nLine {i}: {line.strip()}")
            # Show context (2 lines before and after)
            start = max(0, i-3)
            end = min(len(lines), i+2)
            print("Context:")
            for j in range(start, end):
                marker = ">>>" if j == i-1 else "   "
                print(f"{marker} {j+1}: {lines[j].rstrip()}")

if __name__ == "__main__":
    success = apply_complete_hotfix()
    
    if success:
        show_affected_lines()
        print("\n" + "="*70)
        print("You can now run:")
        print("  python auto_hyperparameter_comprehensive.py")
        print("="*70)
    else:
        print("\nPlease apply manual fix as shown above.")