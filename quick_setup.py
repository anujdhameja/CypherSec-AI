#!/usr/bin/env python3
"""
Quick Setup Script for Devign Repository
Automates the setup process for new machines
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path
import urllib.request
import zipfile
import shutil

def run_command(cmd, description="", check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"Running: {cmd}")
    print('='*60)
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr and check:
            print("STDERR:", result.stderr)
        
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_prerequisites():
    """Check if required software is installed"""
    print("🔍 Checking prerequisites...")
    
    checks = {
        "Python": ["python", "--version"],
        "Git": ["git", "--version"],
        "Java": ["java", "-version"]
    }
    
    results = {}
    for name, cmd in checks.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {name}: OK")
            results[name] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {name}: Not found or not working")
            results[name] = False
    
    return results

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n🐍 Setting up Python virtual environment...")
    
    if Path("venv").exists():
        print("⚠️ Virtual environment already exists")
        return True
    
    # Create venv
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    print("✅ Virtual environment created")
    print("📝 Note: Activate with 'source venv/bin/activate' (Linux/Mac) or 'venv\\Scripts\\activate' (Windows)")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Determine activation command based on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    if Path("install_requirements.py").exists():
        cmd = f"{pip_cmd} install -r requirements-hyperopt.txt"
        return run_command(cmd, "Installing dependencies with pip")
    else:
        print("❌ install_requirements.py not found")
        return False

def download_joern():
    """Download and setup Joern"""
    print("\n⚙️ Setting up Joern...")
    
    joern_dir = Path("joern")
    joern_cli_dir = joern_dir / "joern-cli"
    
    if joern_cli_dir.exists():
        print("⚠️ Joern already exists")
        return True
    
    # Create joern directory
    joern_dir.mkdir(exist_ok=True)
    
    # Download Joern
    joern_url = "https://github.com/joernio/joern/releases/download/v1.1.1741/joern-cli.zip"
    joern_zip = joern_dir / "joern-cli.zip"
    
    try:
        print(f"📥 Downloading Joern from {joern_url}")
        urllib.request.urlretrieve(joern_url, joern_zip)
        
        print("📂 Extracting Joern...")
        with zipfile.ZipFile(joern_zip, 'r') as zip_ref:
            zip_ref.extractall(joern_dir)
        
        # Remove zip file
        joern_zip.unlink()
        
        # Make executable on Unix systems
        if platform.system() != "Windows":
            joern_executable = joern_cli_dir / "joern"
            if joern_executable.exists():
                os.chmod(joern_executable, 0o755)
        
        print("✅ Joern setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Joern: {e}")
        return False

def create_data_directories():
    """Create required data directories"""
    print("\n📁 Creating data directories...")
    
    directories = [
        "data/raw",
        "data/cpg", 
        "data/tokens",
        "data/input",
        "data/w2v",
        "data/model"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")
    
    return True

def update_config():
    """Update configuration file"""
    print("\n⚙️ Updating configuration...")
    
    config_file = Path("configs.json")
    if not config_file.exists():
        print("❌ configs.json not found")
        return False
    
    try:
        # Read current config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Update Joern path
        joern_cli_path = str(Path("joern/joern-cli").absolute())
        if "create" not in config:
            config["create"] = {}
        config["create"]["joern_cli_dir"] = joern_cli_path
        
        # Backup original
        shutil.copy(config_file, "configs.json.backup")
        
        # Write updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration updated")
        print(f"   Joern path: {joern_cli_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def verify_setup():
    """Verify the setup is working"""
    print("\n🔍 Verifying setup...")
    
    # Check if test script exists and run it
    if Path("test_environment.py").exists():
        if platform.system() == "Windows":
            python_cmd = "venv\\Scripts\\python"
        else:
            python_cmd = "venv/bin/python"
        
        return run_command(f"{python_cmd} test_environment.py", "Running environment test")
    else:
        print("⚠️ test_environment.py not found, skipping verification")
        return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*80)
    print("🎉 SETUP COMPLETE!")
    print("="*80)
    
    print("\n📋 Next Steps:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Add your dataset:")
    print("   - Place your dataset file in data/raw/")
    print("   - Ensure it has 'func' and 'target' columns")
    
    print("\n3. Run the pipeline:")
    print("   python main.py --create    # Generate CPGs")
    print("   python main.py --embed     # Create embeddings") 
    print("   python main.py --process   # Train model")
    
    print("\n4. Optional - Hyperparameter optimization:")
    print("   python auto_hyperparameter_bayesian.py")
    print("   python auto_hyperparameter_optuna.py")
    
    print("\n📚 Documentation:")
    print("   - SETUP_GUIDE.md - Complete setup guide")
    print("   - REQUIREMENTS_README.md - Dependency information")
    print("   - README.md - Project overview")
    
    print("\n🆘 If you encounter issues:")
    print("   - Check SETUP_GUIDE.md for troubleshooting")
    print("   - Run: python test_environment.py")
    print("   - Run: python config_verifier.py")

def main():
    """Main setup function"""
    print("🚀 DEVIGN QUICK SETUP")
    print("="*80)
    print("This script will set up the Devign repository on your machine")
    print("="*80)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Error: main.py not found!")
        print("Please run this script from the Devign repository root directory")
        sys.exit(1)
    
    # Check prerequisites
    prereqs = check_prerequisites()
    missing = [name for name, status in prereqs.items() if not status]
    
    if missing:
        print(f"\n❌ Missing prerequisites: {', '.join(missing)}")
        print("Please install them first. See SETUP_GUIDE.md for instructions.")
        sys.exit(1)
    
    # Run setup steps
    steps = [
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up Joern", download_joern),
        ("Creating data directories", create_data_directories),
        ("Updating configuration", update_config),
        ("Verifying setup", verify_setup)
    ]
    
    failed_steps = []
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"📋 Step: {step_name}")
        print('='*60)
        
        if not step_func():
            failed_steps.append(step_name)
            print(f"❌ Failed: {step_name}")
        else:
            print(f"✅ Completed: {step_name}")
    
    # Summary
    if failed_steps:
        print(f"\n❌ Setup completed with errors in: {', '.join(failed_steps)}")
        print("Check the error messages above and refer to SETUP_GUIDE.md")
    else:
        print_next_steps()

if __name__ == "__main__":
    main()