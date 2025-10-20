#!/usr/bin/env python3
"""
Installation script for Devign project dependencies
Handles different installation modes and checks for compatibility
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    if description:
        print(f"📦 {description}")
    print(f"Running: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ is required!")
        return False
    
    print("✅ Python version is compatible")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🔥 Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️ CUDA not available - will use CPU")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet - will check after installation")
        return None

def install_requirements(mode="core"):
    """Install requirements based on mode"""
    requirements_files = {
        "core": "requirements-core.txt",
        "hyperopt": "requirements-hyperopt.txt", 
        "dev": "requirements-dev.txt",
        "full": "requirements.txt"
    }
    
    if mode not in requirements_files:
        print(f"❌ Unknown mode: {mode}")
        return False
    
    req_file = requirements_files[mode]
    
    if not Path(req_file).exists():
        print(f"❌ Requirements file not found: {req_file}")
        return False
    
    # Upgrade pip first
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    cmd = f"pip install -r {req_file}"
    return run_command(cmd, f"Installing {mode} requirements from {req_file}")

def verify_installation():
    """Verify key packages are installed correctly"""
    print("\n🔍 Verifying installation...")
    
    packages_to_check = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("sklearn", "scikit-learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("gensim", "Gensim"),
    ]
    
    failed_packages = []
    
    for package, name in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            failed_packages.append(name)
    
    if failed_packages:
        print(f"\n❌ Failed packages: {', '.join(failed_packages)}")
        return False
    
    print("\n✅ All core packages verified!")
    return True

def create_test_script():
    """Create a simple test script to verify everything works"""
    test_script = '''#!/usr/bin/env python3
"""
Quick test script to verify Devign environment
"""

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torch_geometric
        print(f"✅ PyTorch Geometric {torch_geometric.__version__}")
        
        import sklearn
        print(f"✅ scikit-learn {sklearn.__version__}")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import gensim
        print(f"✅ Gensim {gensim.__version__}")
        
        # Test optional packages
        try:
            import skopt
            print(f"✅ scikit-optimize {skopt.__version__}")
        except ImportError:
            print("⚠️ scikit-optimize not installed (optional)")
        
        try:
            import optuna
            print(f"✅ Optuna {optuna.__version__}")
        except ImportError:
            print("⚠️ Optuna not installed (optional)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_torch_geometric():
    """Test PyTorch Geometric functionality"""
    print("\\nTesting PyTorch Geometric...")
    
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
        
        # Create simple test data
        x = torch.randn(4, 2)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        
        # Create simple model
        conv = GCNConv(2, 2)
        out = conv(data.x, data.edge_index)
        
        print("✅ PyTorch Geometric working correctly")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch Geometric test failed: {e}")
        return False

def main():
    print("="*60)
    print("🧪 DEVIGN ENVIRONMENT TEST")
    print("="*60)
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_torch_geometric():
        success = False
    
    print("\\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED! Environment is ready.")
    else:
        print("❌ Some tests failed. Check the errors above.")
    print("="*60)
    
    return success

if __name__ == "__main__":
    main()
'''
    
    with open("test_environment.py", "w") as f:
        f.write(test_script)
    
    print("✅ Created test_environment.py")

def main():
    parser = argparse.ArgumentParser(description="Install Devign project dependencies")
    parser.add_argument(
        "--mode", 
        choices=["core", "hyperopt", "dev", "full"],
        default="core",
        help="Installation mode (default: core)"
    )
    parser.add_argument(
        "--skip-checks", 
        action="store_true",
        help="Skip compatibility checks"
    )
    
    args = parser.parse_args()
    
    print("🚀 DEVIGN PROJECT SETUP")
    print("="*60)
    print(f"Installation mode: {args.mode}")
    print("="*60)
    
    # Check Python version
    if not args.skip_checks and not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements(args.mode):
        print("\n❌ Installation failed!")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Verification failed!")
        sys.exit(1)
    
    # Check CUDA after installation
    if not args.skip_checks:
        check_cuda()
    
    # Create test script
    create_test_script()
    
    print("\n" + "="*60)
    print("🎉 INSTALLATION COMPLETE!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Run: python test_environment.py")
    print("2. If tests pass, you're ready to use Devign!")
    print("\n📚 Available modes:")
    print("  - core: Basic functionality")
    print("  - hyperopt: + Hyperparameter optimization")
    print("  - dev: + Development tools")
    print("  - full: Everything")
    print("\n🔄 To upgrade: python install_requirements.py --mode <mode>")

if __name__ == "__main__":
    main()