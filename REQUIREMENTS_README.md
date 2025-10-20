# Devign Project Requirements

This document explains the dependency management for the Devign project.

## 📋 Requirements Files

### Core Files
- **`requirements.txt`** - Complete dependencies with exact versions
- **`requirements-core.txt`** - Minimal dependencies for basic functionality
- **`requirements-hyperopt.txt`** - Adds hyperparameter optimization libraries
- **`requirements-dev.txt`** - Adds development and testing tools

### Installation Scripts
- **`install_requirements.py`** - Automated installation with compatibility checks
- **`test_environment.py`** - Verify installation (auto-generated)

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)
```bash
# Basic installation
python install_requirements.py --mode core

# With hyperparameter optimization
python install_requirements.py --mode hyperopt

# Full development environment
python install_requirements.py --mode dev
```

### Option 2: Manual Installation
```bash
# Basic installation
pip install -r requirements-core.txt

# With hyperparameter optimization
pip install -r requirements-hyperopt.txt

# Full development environment
pip install -r requirements-dev.txt
```

## 📦 Key Dependencies

### Core Libraries
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.8.0 | Deep learning framework |
| `torch-geometric` | 2.6.1 | Graph neural networks |
| `scikit-learn` | 1.7.1 | Machine learning utilities |
| `pandas` | 2.3.2 | Data manipulation |
| `numpy` | 1.26.4 | Numerical computing |
| `gensim` | 4.3.3 | Word2Vec embeddings |
| `cpgclientlib` | 0.11.321 | Code Property Graph processing |

### Hyperparameter Optimization
| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-optimize` | 0.10.2 | Bayesian optimization |
| `optuna` | >=3.6.0 | Advanced hyperparameter tuning |
| `plotly` | >=5.17.0 | Visualization for results |

### Development Tools
| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >=7.4.0 | Testing framework |
| `black` | >=23.0.0 | Code formatting |
| `flake8` | >=6.0.0 | Linting |
| `jupyter` | >=1.0.0 | Interactive development |

## 🔧 System Requirements

### Python Version
- **Required:** Python 3.8+
- **Recommended:** Python 3.9 or 3.10
- **Tested:** Python 3.10

### Hardware Requirements
- **RAM:** Minimum 8GB, Recommended 16GB+
- **Storage:** 5GB+ free space
- **GPU:** Optional but recommended (CUDA-compatible)

### Operating System
- **Windows:** 10/11 (tested)
- **Linux:** Ubuntu 18.04+ (should work)
- **macOS:** 10.15+ (should work)

## 🐛 Troubleshooting

### Common Issues

#### 1. PyTorch Geometric Installation
If you encounter issues with `torch-geometric`:
```bash
# Install PyTorch first
pip install torch==2.8.0

# Then install PyTorch Geometric
pip install torch-geometric==2.6.1
```

#### 2. CUDA Issues
For GPU support:
```bash
# Check CUDA version
nvidia-smi

# Install CUDA-compatible PyTorch
pip install torch==2.8.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3. Gensim Compilation Issues
On Windows, if Gensim fails to compile:
```bash
# Install pre-compiled version
pip install --only-binary=gensim gensim==4.3.3
```

#### 4. Memory Issues
If you encounter memory errors:
- Reduce batch size in configs
- Use CPU instead of GPU for small datasets
- Close other applications

### Verification Steps

1. **Test imports:**
   ```bash
   python test_environment.py
   ```

2. **Test basic functionality:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
   ```

3. **Test GPU (if available):**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 🔄 Updating Dependencies

### Update All Packages
```bash
pip install --upgrade -r requirements.txt
```

### Update Specific Package
```bash
pip install --upgrade torch torch-geometric
```

### Check for Outdated Packages
```bash
pip list --outdated
```

## 📊 Installation Modes Comparison

| Feature | Core | HyperOpt | Dev | Full |
|---------|------|----------|-----|------|
| Basic training | ✅ | ✅ | ✅ | ✅ |
| Bayesian optimization | ❌ | ✅ | ✅ | ✅ |
| Optuna optimization | ❌ | ✅ | ✅ | ✅ |
| Visualization | ❌ | ✅ | ✅ | ✅ |
| Testing tools | ❌ | ❌ | ✅ | ✅ |
| Code formatting | ❌ | ❌ | ✅ | ✅ |
| Jupyter notebooks | ❌ | ❌ | ✅ | ✅ |
| All extras | ❌ | ❌ | ❌ | ✅ |

## 🆘 Getting Help

If you encounter issues:

1. **Check this document** for common solutions
2. **Run diagnostics:** `python install_requirements.py --mode core`
3. **Test environment:** `python test_environment.py`
4. **Check logs** for specific error messages
5. **Create an issue** with full error output

## 📝 Notes

- All version numbers are tested and known to work together
- Using exact versions ensures reproducibility
- Optional dependencies are marked clearly
- Development tools are separated to keep core installation lightweight

## 🔗 Useful Links

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [PyTorch Geometric Installation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [scikit-optimize Documentation](https://scikit-optimize.github.io/stable/)