#!/bin/bash
# Linux/macOS Setup Script for Devign Repository
# Run this after cloning the repository

set -e  # Exit on any error

echo "========================================"
echo "DEVIGN REPOSITORY SETUP - LINUX/MACOS"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found!"
    echo "Please run this script from the Devign repository root directory"
    exit 1
fi

echo "Step 1: Creating virtual environment..."
python3 -m venv venv

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Upgrading pip..."
python -m pip install --upgrade pip

echo "Step 4: Installing dependencies..."
pip install -r requirements-hyperopt.txt

echo "Step 5: Creating data directories..."
mkdir -p data/{raw,cpg,tokens,input,w2v,model}

echo "Step 6: Testing environment..."
python test_environment.py || echo "WARNING: Environment test failed"

echo "========================================"
echo "SETUP COMPLETE!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download and extract Joern to joern/joern-cli/"
echo "2. Update configs.json with correct Joern path"
echo "3. Place your dataset in data/raw/"
echo "4. Run: python main.py --create --embed --process"
echo ""
echo "To activate environment in future sessions:"
echo "source venv/bin/activate"
echo ""