# ✅ Devign Setup Checklist - New Machine

## 🚀 Quick Setup (Automated)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd devign

# 2. Run automated setup
python quick_setup.py

# 3. Activate environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 4. Add your dataset to data/raw/

# 5. Run pipeline
python main.py --create --embed --process
```

## 📋 Manual Setup Checklist

### Prerequisites ✅
- [ ] Git installed and working
- [ ] Python 3.8+ installed
- [ ] Java 11+ installed
- [ ] 10GB+ free disk space
- [ ] Stable internet connection

### Repository Setup ✅
- [ ] Repository cloned: `git clone <repo-url>`
- [ ] Navigate to directory: `cd devign`
- [ ] Verify files present: `ls -la` (should see main.py, configs.json, etc.)

### Python Environment ✅
- [ ] Virtual environment created: `python -m venv venv`
- [ ] Environment activated: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- [ ] Dependencies installed: `python install_requirements.py --mode hyperopt`
- [ ] Installation verified: `python test_environment.py`

### Joern Setup ✅
- [ ] Joern downloaded and extracted to `joern/joern-cli/`
- [ ] Joern executable: `./joern/joern-cli/joern --version`
- [ ] Config updated with Joern path in `configs.json`

### Data Directories ✅
- [ ] Created: `mkdir -p data/{raw,cpg,tokens,input,w2v,model}`
- [ ] Verified structure: `ls -la data/`

### Configuration ✅
- [ ] `configs.json` exists and is valid
- [ ] Joern path updated in config
- [ ] Paths point to correct directories
- [ ] Config verified: `python config_verifier.py`

### Dataset ✅
- [ ] Dataset file placed in `data/raw/`
- [ ] Dataset has required columns: `func`, `target`
- [ ] Dataset format verified

### Pipeline Execution ✅
- [ ] Create step: `python main.py --create` ✅
- [ ] Embed step: `python main.py --embed` ✅  
- [ ] Process step: `python main.py --process` ✅
- [ ] No errors in console output
- [ ] Model files created in `data/model/`

### Verification ✅
- [ ] Environment test passes: `python test_environment.py`
- [ ] Data diagnostic runs: `python data_diagnostic.py`
- [ ] Model can be loaded without errors
- [ ] Basic training completes successfully

### Optional Features ✅
- [ ] Bayesian optimization: `python auto_hyperparameter_bayesian.py`
- [ ] Optuna optimization: `python auto_hyperparameter_optuna.py`
- [ ] GPU acceleration working (if available)

## 🚨 Common Issues & Solutions

### ❌ Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements-hyperopt.txt
```

### ❌ Joern Not Found
```bash
# Check Joern installation
ls -la joern/joern-cli/
# Update configs.json with correct path
```

### ❌ CUDA Issues
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Install CPU version if needed
pip install torch==2.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### ❌ Memory Issues
```bash
# Reduce batch size in configs.json
# Monitor memory: python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available/1024**3:.1f}GB')"
```

## 📞 Getting Help

1. **Check console output** for specific error messages
2. **Run diagnostics:** `python data_diagnostic.py`
3. **Verify environment:** `python test_environment.py`
4. **Check configuration:** `python config_verifier.py`
5. **Review SETUP_GUIDE.md** for detailed troubleshooting

## 🎯 Success Indicators

✅ **Setup Complete When:**
- All checklist items marked ✅
- `python test_environment.py` passes
- Pipeline runs without errors
- Model training completes successfully
- Results saved in appropriate directories

## ⏱️ Estimated Time

- **Automated setup:** 15-30 minutes
- **Manual setup:** 30-60 minutes
- **First pipeline run:** 30-120 minutes (depends on dataset size)

## 🎉 Ready to Go!

Once all items are checked ✅, your Devign environment is ready for:
- Code vulnerability detection
- Model training and evaluation
- Hyperparameter optimization
- Research and development

**Happy coding! 🚀**