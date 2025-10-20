import sys
import os

print("="*70)
print("PYTHON ENVIRONMENT TEST")
print("="*70)
print(f"Python executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Try to create a file to verify write permissions
try:
    with open('test_write.txt', 'w') as f:
        f.write('Test write successful')
    print("\n✓ Successfully wrote to test file")
    os.remove('test_write.txt')
    print("✓ Successfully removed test file")
except Exception as e:
    print(f"\n❌ Failed to write test file: {e}")

print("\nEnvironment test complete!")
