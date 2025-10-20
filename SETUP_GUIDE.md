# 🚀 Devign Repository Setup Guide - New Machine Installation

## 📋 Complete Step-by-Step Setup Process

This guide will walk you through setting up the Devign repository on a new machine from scratch.

## 🔧 Prerequisites

### System Requirements
- **OS:** Windows 10/11, Linux (Ubuntu 18.04+), or macOS 10.15+
- **RAM:** Minimum 8GB, Recommended 16GB+
- **Storage:** 10GB+ free space
- **Internet:** Stable connection for downloads

### Required Software
1. **Git** (for cloning repository)
2. **Python 3.8+** (recommended: Python 3.10)
3. **Java 11+** (for Joern CPG processing)

---

## 📥 STEP 1: Install Prerequisites

### Windows:
```powershell
# Install Git
winget install Git.Git

# Install Python (if not installed)
winget install Python.Python.3.10

# Install Java
winget install Microsoft.OpenJDK.11

# Verify installations
git --version
python --version
java -version
```

### Linux (Ubuntu/Debian):
```bash
# Update package list
sudo apt update

# Install Git, Python, Java
sudo apt install git python3.10 python3.10-venv python3-pip openjdk-11-jdk

# Verify installations
git --version
python3 --version
java -version
```

### macOS:
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install prerequisites
brew install git python@3.10 openjdk@11

# Verify installations
git --version
python3 --version
java -version
```

---

## 📂 STEP 2: Clone Repository

```bash
# Navigate to desired directory
cd /path/to/your/projects  # Change this to your preferred location

# Clone the repository
git clone <YOUR_REPOSITORY_URL>  # Replace with your actual repo URL

# Navigate into the project
cd devign

# Verify repository structure
ls -la
```

**Expected structure:**
```
devign/
├── src/
├── data/
├── configs.json
├── main.py
├── requirements*.txt
├── install_requirements.py
└── README.md
```

---

## 🐍 STEP 3: Set Up Python Environment

### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Verify activation (should show (venv) in prompt)
which python  # Should point to venv/bin/python
```

### Install Dependencies
```bash
# Option 1: Automated installation (RECOMMENDED)
python install_requirements.py --mode hyperopt

# Option 2: Manual installation
pip install -r requirements-hyperopt.txt

# Verify installation
python test_environment.py
```

---

## ⚙️ STEP 4: Configure Joern (CPG Processing)

### Download Joern
```bash
# Create joern directory
mkdir -p joern

# Download Joern (adjust version as needed)
cd joern
wget https://github.com/joernio/joern/releases/download/v1.1.1741/joern-cli.zip

# Extract
unzip joern-cli.zip

# Make executable (Linux/macOS)
chmod +x joern-cli/joern

# Test Joern
./joern-cli/joern --version

# Go back to project root
cd ..
```

### Update Joern Path in Config
```bash
# Edit configs.json to set correct Joern path
# Update the "joern_cli_dir" field to point to your joern-cli directory
```

---

## 📊 STEP 5: Prepare Data Directory Structure

```bash
# Create required data directories
mkdir -p data/{raw,cpg,tokens,input,w2v,model}

# Verify structure
tree data/  # or ls -la data/ if tree not available
```

**Expected data structure:**
```
data/
├── raw/          # Original dataset files
├── cpg/          # Code Property Graphs
├── tokens/       # Tokenized data
├── input/        # Processed input for training
├── w2v/          # Word2Vec models
└── model/        # Trained models
```

---

## 🔧 STEP 6: Configuration Setup

### Update configs.json
```bash
# Copy and edit the configuration file
cp configs.json configs.json.backup

# Edit configs.json with your paths
# Update these fields:
```

```json
{
  "create": {
    "joern_cli_dir": "/absolute/path/to/joern-cli"
  },
  "paths": {
    "raw": "data/raw",
    "cpg": "data/cpg", 
    "tokens": "data/tokens",
    "input": "data/input",
    "w2v": "data/w2v",
    "model": "data/model",
    "joern": "joern"
  }
}
```

### Verify Configuration
```bash
# Test configuration
python config_verifier.py
```

---

## 📥 STEP 7: Data Preparation

### Option 1: Use Sample Data (for testing)
```bash
# Download sample dataset (if available)
# Place your dataset files in data/raw/

# Example structure:
# data/raw/dataset.csv  # Your vulnerability dataset
```

### Option 2: Use Your Own Dataset
```bash
# Copy your dataset to data/raw/
cp /path/to/your/dataset.csv data/raw/

# Ensure dataset has required columns:
# - func: function code
# - target: vulnerability label (0/1)
# - project: project name (optional)
```

---

## 🏃‍♂️ STEP 8: Run the Pipeline

### Full Pipeline Execution
```bash
# Activate virtual environment (if not already active)
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Step 1: Create CPG files from source code
python main.py --create

# Step 2: Generate embeddings and process data
python main.py --embed

# Step 3: Train the model
python main.py --process

# Or run with early stopping
python main.py --process_stopping
```

### Individual Steps (if needed)
```bash
# Create CPGs only
python main.py -c

# Generate embeddings only  
python main.py -e

# Train model only
python main.py -p

# Train with early stopping
python main.py -pS
```

---

## 🔍 STEP 9: Verification and Testing

### Test Basic Functionality
```bash
# Run environment test
python test_environment.py

# Run data diagnostics
python data_diagnostic.py

# Test model loading
python -c "
import torch
from src.process.balanced_training_config import BalancedDevignModel
model = BalancedDevignModel(input_dim=100, output_dim=2)
print('✅ Model creation successful')
"
```

### Run Hyperparameter Optimization (Optional)
```bash
# Bayesian optimization
python auto_hyperparameter_bayesian.py

# Optuna optimization  
python auto_hyperparameter_optuna.py

# Quick test (1 trial)
python test_bayesian_quick.py
```

---

## 🚨 STEP 10: Troubleshooting Common Issues

### Issue 1: Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements-hyperopt.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Issue 2: Joern Not Found
```bash
# Check Joern installation
ls -la joern/joern-cli/
./joern/joern-cli/joern --version

# Update configs.json with correct path
```

### Issue 3: CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CPU-only version if needed
pip install torch==2.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Issue 4: Memory Issues
```bash
# Reduce batch size in configs.json
# Set "batch_size": 4 instead of 32

# Monitor memory usage
python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

---

## 📋 STEP 11: Verification Checklist

### ✅ Pre-flight Checklist
- [ ] Git repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`python test_environment.py` passes)
- [ ] Joern installed and configured
- [ ] Data directories created
- [ ] configs.json updated with correct paths
- [ ] Dataset placed in data/raw/

### ✅ Pipeline Checklist  
- [ ] Create step completed (CPG files generated)
- [ ] Embed step completed (embeddings created)
- [ ] Process step completed (model trained)
- [ ] No error messages in logs
- [ ] Model files created in data/model/

### ✅ Optional Features
- [ ] Hyperparameter optimization working
- [ ] GPU acceleration enabled (if available)
- [ ] Visualization tools working

---

## 🎯 Quick Start Commands (Summary)

```bash
# 1. Clone and setup
git clone <repo-url>
cd devign
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
python install_requirements.py --mode hyperopt

# 3. Setup Joern (download and extract to joern/ directory)
# Update configs.json with correct joern_cli_dir path

# 4. Create data directories
mkdir -p data/{raw,cpg,tokens,input,w2v,model}

# 5. Add your dataset to data/raw/

# 6. Run pipeline
python main.py --create
python main.py --embed  
python main.py --process

# 7. Verify
python test_environment.py
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check logs** in the console output
2. **Run diagnostics:** `python data_diagnostic.py`
3. **Verify environment:** `python test_environment.py`
4. **Check configuration:** `python config_verifier.py`
5. **Review this guide** for missed steps

## 🎉 Success!

Once all steps are completed, you should have a fully functional Devign environment ready for:
- ✅ Code vulnerability detection
- ✅ Model training and evaluation  
- ✅ Hyperparameter optimization
- ✅ Research and development

**Happy coding! 🚀**