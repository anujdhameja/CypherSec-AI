"""
Verify that the hotfix was applied correctly
Run: python verify_hotfix.py
"""

import re

def verify_hotfix():
    """Verify the hotfix was applied"""
    
    filepath = 'auto_hyperparameter_comprehensive.py'
    
    print("="*70)
    print("VERIFYING HOTFIX")
    print("="*70)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = f.readlines()
    
    # Count .long() conversions
    long_count = content.count('targets.long()')
    print(f"\n1. Found {long_count} target.long() conversions")
    
    if long_count >= 2:
        print("   ✓ At least 2 conversions found (train and validate)")
    else:
        print(f"   ❌ Only {long_count} conversion(s) found - need at least 2")
    
    # Find specific locations
    print("\n2. Checking specific locations:")
    
    train_epoch_fixed = False
    validate_epoch_fixed = False
    
    in_train_epoch = False
    in_validate_epoch = False
    
    for i, line in enumerate(lines):
        # Track which method we're in
        if 'def train_epoch(' in line:
            in_train_epoch = True
            in_validate_epoch = False
        elif 'def validate_epoch(' in line:
            in_validate_epoch = True
            in_train_epoch = False
        elif line.strip().startswith('def '):
            in_train_epoch = False
            in_validate_epoch = False
        
        # Check for .long() conversion
        if 'targets.long()' in line or 'targets = targets.long()' in line:
            if in_train_epoch and not train_epoch_fixed:
                print(f"   ✓ train_epoch fixed at line {i+1}")
                train_epoch_fixed = True
            elif in_validate_epoch and not validate_epoch_fixed:
                print(f"   ✓ validate_epoch fixed at line {i+1}")
                validate_epoch_fixed = True
    
    if not train_epoch_fixed:
        print("   ❌ train_epoch NOT fixed")
    if not validate_epoch_fixed:
        print("   ❌ validate_epoch NOT fixed")
    
    # Check for problematic patterns (targets used without .long())
    print("\n3. Checking for remaining issues:")
    
    # Pattern: "targets = batch.y" followed by "criterion(outputs, targets)" without .long()
    pattern = r'targets\s*=\s*(?:batch\.y|batch\[1\])[^\n]*\n(?:.*\n){0,5}?.*criterion\(outputs,\s*targets\)'
    
    matches = list(re.finditer(pattern, content))
    issues_found = 0
    
    for match in matches:
        match_text = match.group(0)
        if 'targets.long()' not in match_text and '.long()' not in match_text:
            issues_found += 1
            # Find line number
            line_num = content[:match.start()].count('\n') + 1
            print(f"   ⚠️  Potential issue at line ~{line_num}")
    
    if issues_found == 0:
        print("   ✓ No remaining issues found")
    else:
        print(f"   ❌ Found {issues_found} potential issue(s)")
    
    # Final verdict
    print("\n" + "="*70)
    print("VERDICT:")
    print("="*70)
    
    if train_epoch_fixed and validate_epoch_fixed and issues_found == 0:
        print("✅ HOTFIX VERIFIED - Ready to run!")
        print("\nYou can now run:")
        print("  python auto_hyperparameter_comprehensive.py")
        print("\nThe target dtype error should be fixed.")
        return True
    else:
        print("❌ HOTFIX INCOMPLETE")
        print("\nIssues:")
        if not train_epoch_fixed:
            print("  - train_epoch not fixed")
        if not validate_epoch_fixed:
            print("  - validate_epoch not fixed")
        if issues_found > 0:
            print(f"  - {issues_found} remaining issue(s)")
        
        print("\nTry running:")
        print("  python hotfix_v2_complete.py")
        return False

def show_manual_fix_instructions():
    """Show manual fix instructions"""
    print("\n" + "="*70)
    print("MANUAL FIX INSTRUCTIONS")
    print("="*70)
    print("""
If automated fix doesn't work, manually edit auto_hyperparameter_comprehensive.py:

1. Find the train_epoch method (around line 320-350)
   Look for:
   ```python
   targets = batch.y
   loss = criterion(outputs, targets)
   ```
   
   Change to:
   ```python
   targets = batch.y
   targets = targets.long()  # ADD THIS LINE
   loss = criterion(outputs, targets)
   ```

2. Find the validate_epoch method (around line 370-400)
   Look for:
   ```python
   targets = batch.y
   loss = criterion(outputs, targets)
   ```
   
   Change to:
   ```python
   targets = batch.y
   targets = targets.long()  # ADD THIS LINE
   loss = criterion(outputs, targets)
   ```

Save the file and run verify_hotfix.py again.
""")

if __name__ == "__main__":
    success = verify_hotfix()
    
    if not success:
        show_manual_fix_instructions()
    
    print("\n" + "="*70)