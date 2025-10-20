# 🚀 NEW MACHINE SETUP - Complete Guide

## 🎯 Overview

This guide provides **3 different methods** to set up the Devign repository on a new machine, from beginner-friendly to advanced.

---

## 🚀 METHOD 1: Automated Setup (RECOMMENDED)

### Prerequisites
- Git, Python 3.8+, Java 11+ installed
- Internet connection

### Steps
```bash
# 1. Clone repository
git clone <YOUR_REPOSITORY_URL>
cd devign

# 2. Run automated setup
python quick_setup.py

# 3. Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 4. Add dataset and run
# Place your dataset in data/raw/
python main.py --create --embed --process
```

**Time:** ~15-30 minutes

---

## 🛠️ METHOD 2: Script-Based Setup

### Windows
```batch
# 1. Clone repository
git clone <YOUR_REPOSITORY_URL>
cd devign

# 2. Run setup script
setup.bat

# 3. Manual Joern setup (see below)
# 4. Add dataset and run pipeline
```

### Linux/macOS
```bash
# 1. Clone repository
git clone <YOUR_REPOSITORY_URL>
cd devign

# 2. Make script executable and run
chmod +x setup.sh
./setup.sh

# 3. Manual Joern setup (see below)
# 4. Add dataset and run pipeline
```

**Time:** ~20-40 minutes

---

## 📋 METHOD 3: Manual Setup (Step-by-Step)

### Step 1: Prerequisites Installation

#### Windows
```powershell
# Install via winget (Windows Package Manager)
winget install Git.Git
winget install Python.Python.3.10
winget install Microsoft.OpenJDK.11

# Verify
git --version
python --version
java -version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install git python3.10 python3.10-venv python3-pip openjdk-11-jdk

# Verify
git --version
python3 --version
java -version
```

#### macOS
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install packages
brew install git python@3.10 openjdk@11

# Verify
git --version
python3 --version
java -version
```

### Step 2: Repository Setup
```bash
# Clone repository
git clone <YOUR_REPOSITORY_URL>
cd devign

# Verify structure
ls -la  # Should see main.py, configs.json, etc.
```

### Step 3: Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
python install_requirements.py --mode hyperopt
# OR manually:
pip install -r requirements-hyperopt.txt

# Verify installation
python test_environment.py
```

### Step 4: Joern Setup
```bash
# Create joern directory
mkdir joern
cd joern

# Download Joern (adjust URL for latest version)
wget https://github.com/joernio/joern/releases/download/v1.1.1741/joern-cli.zip
# OR on Windows: download manually from GitHub

# Extract
unzip joern-cli.zip  # Linux/Mac
# OR use Windows built-in extraction

# Make executable (Linux/Mac only)
chmod +x joern-cli/joern

# Test
./joern-cli/joern --version  # Linux/Mac
# OR
joern-cli\joern.bat --version  # Windows

# Go back to project root
cd ..
```

### Step 5: Configuration
```bash
# Create data directories
mkdir -p data/{raw,cpg,tokens,input,w2v,model}  # Linux/Mac
# OR Windows:
mkdir data\raw data\cpg data\tokens data\input data\w2v data\model

# Update configs.json
# Edit the "joern_cli_dir" field to point to your joern-cli directory
# Example: "/absolute/path/to/devign/joern/joern-cli"
```

### Step 6: Dataset Preparation
```bash
# Place your dataset in data/raw/
# Ensure it has columns: 'func' (code), 'target' (0/1 labels)

# Verify dataset
python data_diagnostic.py
```

### Step 7: Run Pipeline
```bash
# Full pipeline
python main.py --create    # Generate CPGs (~30-60 min)
python main.py --embed     # Create embeddings (~15-30 min)
python main.py --process   # Train model (~15-45 min)

# OR all at once
python main.py --create --embed --process
```

**Time:** ~60-90 minutes

---

## 🔧 Joern Manual Setup (Detailed)

### Download Options
1. **Direct Download:** https://github.com/joernio/joern/releases
2. **Latest Version:** Look for `joern-cli.zip` in releases
3. **Recommended:** v1.1.1741 or newer

### Installation Steps
```bash
# 1. Create directory
mkdir joern && cd joern

# 2. Download (choose one method)
# Method A: wget (Linux/Mac)
wget https://github.com/joernio/joern/releases/download/v1.1.1741/joern-cli.zip

# Method B: curl (Linux/Mac)
curl -L -o joern-cli.zip https://github.com/joernio/joern/releases/download/v1.1.1741/joern-cli.zip

# Method C: Manual download (Windows)
# Download from GitHub releases page

# 3. Extract
unzip joern-cli.zip  # Linux/Mac
# OR use Windows extraction

# 4. Test installation
./joern-cli/joern --version  # Linux/Mac
joern-cli\joern.bat --version  # Windows

# 5. Update configs.json
# Set "joern_cli_dir" to full path of joern-cli directory
```

---

## 📊 Pipeline Execution Guide

### Sequential Execution (Recommended)
```bash
# Step 1: Create CPGs from source code
python main.py --create
# Expected: CPG files in data/cpg/

# Step 2: Generate embeddings
python main.py --embed  
# Expected: Word2Vec model in data/w2v/, input files in data/input/

# Step 3: Train model
python main.py --process
# Expected: Trained model in data/model/
```

### Combined Execution
```bash
# All steps at once
python main.py --create --embed --process

# With early stopping
python main.py --create --embed --process_stopping
```

### Individual Steps (if needed)
```bash
python main.py -c   # Create only
python main.py -e   # Embed only  
python main.py -p   # Process only
python main.py -pS  # Process with early stopping
```

---

## 🔍 Verification & Testing

### Environment Verification
```bash
# Test all imports and dependencies
python test_environment.py

# Check configuration
python config_verifier.py

# Diagnose data
python data_diagnostic.py
```

### Pipeline Verification
```bash
# Check CPG generation
ls -la data/cpg/  # Should contain .pkl files

# Check embeddings
ls -la data/w2v/  # Should contain w2v.model

# Check input data
ls -la data/input/  # Should contain *_input.pkl files

# Check trained model
ls -la data/model/  # Should contain devign.model
```

### Optional: Hyperparameter Optimization
```bash
# Bayesian optimization
python auto_hyperparameter_bayesian.py

# Optuna optimization
python auto_hyperparameter_optuna.py

# Quick test (1 trial)
python test_bayesian_quick.py
```

---

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements-hyperopt.txt

# Check virtual environment
which python  # Should point to venv/bin/python
```

#### 2. Joern Not Found
```bash
# Check Joern installation
ls -la joern/joern-cli/
./joern/joern-cli/joern --version

# Update configs.json with absolute path
pwd  # Get current directory
# Update "joern_cli_dir": "/full/path/to/joern/joern-cli"
```

#### 3. CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CPU version if needed
pip install torch==2.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

#### 4. Memory Issues
```bash
# Check available memory
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"

# Reduce batch size in configs.json
# Change "batch_size" from 32 to 8 or 4
```

#### 5. Dataset Issues
```bash
# Check dataset format
python -c "
import pandas as pd
df = pd.read_csv('data/raw/your_dataset.csv')
print(df.columns.tolist())
print(df.head())
"

# Required columns: 'func', 'target'
```

---

## 📋 Success Checklist

### ✅ Setup Complete When:
- [ ] All prerequisites installed
- [ ] Repository cloned successfully
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`python test_environment.py` passes)
- [ ] Joern downloaded and configured
- [ ] Data directories created
- [ ] configs.json updated with correct paths
- [ ] Dataset placed in data/raw/

### ✅ Pipeline Complete When:
- [ ] Create step completed (CPG files in data/cpg/)
- [ ] Embed step completed (w2v model in data/w2v/)
- [ ] Process step completed (model in data/model/)
- [ ] No error messages in console
- [ ] Model can be loaded successfully

---

## 🎯 Expected Timeline

| Task | Time Estimate |
|------|---------------|
| Prerequisites installation | 10-20 min |
| Repository setup | 5-10 min |
| Python environment | 10-15 min |
| Joern setup | 5-10 min |
| Configuration | 5 min |
| **Total Setup** | **35-60 min** |
| Create step (CPG) | 30-60 min |
| Embed step | 15-30 min |
| Process step (training) | 15-45 min |
| **Total Pipeline** | **60-135 min** |

*Times vary based on dataset size and hardware*

---

## 🎉 Success!

Once setup is complete, you'll have a fully functional Devign environment capable of:

✅ **Code vulnerability detection**
✅ **Graph neural network training**
✅ **Hyperparameter optimization**
✅ **Research and development**

**Ready to detect vulnerabilities! 🚀**

---

## 📞 Need Help?

1. **Check console output** for specific errors
2. **Review troubleshooting section** above
3. **Run diagnostics:** `python data_diagnostic.py`
4. **Verify setup:** `python test_environment.py`
5. **Check detailed guide:** `SETUP_GUIDE.md`