import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*70)
print("TESTING IMPORTS")
print("="*70)
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    print("\nAttempting to import InputDataset...")
    from src.utils.objects.input_dataset import InputDataset
    print("✓ Successfully imported InputDataset")
    
    print("\nAttempting to import Devign...")
    from src.process.devign import Devign
    print("✓ Successfully imported Devign")
    
    print("\n✓ All imports successful!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
