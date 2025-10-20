# Bayesian Hyperparameter Optimization - Fixes Applied

## Summary
Successfully updated `auto_hyperparameter_bayesian.py` to work with the current model architecture and data pipeline.

## CRITICAL FIX APPLIED
**Data Loading Issue:** The original `train_val_test_split()` function had a bug where it tried to pass DataFrames to `InputDataset` constructor, but `InputDataset` expects directory paths. Fixed by creating a custom `DataFrameDataset` class and using sklearn's `train_test_split` for proper data splitting.

## Issues Fixed

### 1. **Outdated Imports**
**Problem:** Using old import paths and modules that no longer exist
```python
# OLD (broken)
from src.data.datamanager import DataManager
from src.process.model import DevignModel
from src.process.modeling import Modeling
from src.utils.objects.input_dataset import InputDataset
from configs import configs
```

**Fixed:** Updated to current module structure
```python
# NEW (working)
import configs
import src.data as data
import src.process as process
from src.process.balanced_training_config import BalancedDevignModel
from src.process.step import Step
from src.process.modeling import Train
from torch_geometric.loader import DataLoader
import pandas as pd
```

### 2. **Incorrect Data Loading & Critical Bug Fix**
**Problem:** Multiple issues with data loading:
1. Trying to load non-existent train/val split files
2. `train_val_test_split()` function had a bug - it passed DataFrames to `InputDataset` constructor, but `InputDataset` expects directory paths

```python
# OLD (broken)
train_ds = InputDataset(dataset_path / 'train.pkl')
val_ds = InputDataset(dataset_path / 'val.pkl')

# Also broken in train_val_test_split():
return InputDataset(train), InputDataset(test), InputDataset(val)  # DataFrames passed to InputDataset!
```

**Fixed:** Created custom `DataFrameDataset` class and proper data splitting
```python
# NEW (working)
class DataFrameDataset:
    def __init__(self, dataframe):
        self.data = dataframe['input'].tolist()
        self.targets = dataframe['target'].tolist()
    
    def get_loader(self, batch_size=32, shuffle=True):
        # Returns proper PyTorch Geometric DataLoader

# Proper splitting with sklearn
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(input_dataset, test_size=0.2, stratify=input_dataset['target'])
train_df, val_df = train_test_split(train_temp, test_size=0.2, stratify=train_temp['target'])

self.train_dataset = DataFrameDataset(train_df)  # 4,725 samples
self.val_dataset = DataFrameDataset(val_df)      # 1,182 samples  
self.test_dataset = DataFrameDataset(test_df)    # 1,477 samples
```

### 3. **Wrong Model Architecture**
**Problem:** Using old DevignModel with outdated configuration
```python
# OLD (broken)
model = DevignModel(
    input_dim=configs['data']['input_dataset']['nodes_dim'],
    output_dim=2,
    model_params=model_config
).to(self.device)
```

**Fixed:** Using current BalancedDevignModel
```python
# NEW (working)
embed_config = configs.Embed()
model = BalancedDevignModel(
    input_dim=embed_config.nodes_dim,  # 100
    output_dim=2,
    hidden_dim=config['hidden_dim'],
    num_steps=config['num_steps'],
    dropout=config['dropout']
).to(self.device)
```

### 4. **Outdated Training Loop**
**Problem:** Using non-existent Modeling class
```python
# OLD (broken)
modeling = Modeling(
    model=model,
    dataset={'train': self.train_dataset, 'val': self.val_dataset},
    params=train_config,
    device=self.device
)
```

**Fixed:** Using current Step-based training approach
```python
# NEW (working)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
step = Step(model=model, loss_function=loss_fn, optimizer=optimizer)

train_loader = self.train_dataset.get_loader(config['batch_size'], shuffle=True)
val_loader = self.val_dataset.get_loader(config['batch_size'], shuffle=False)

train_loader_step = process.LoaderStep("Train", train_loader, self.device)
val_loader_step = process.LoaderStep("Validation", val_loader, self.device)
```

### 5. **Updated Training Execution**
**Problem:** Calling non-existent methods
```python
# OLD (broken)
train_acc, train_loss = modeling.train_epoch(epoch)
val_acc, val_loss = modeling.validate_epoch(epoch)
```

**Fixed:** Using current training approach
```python
# NEW (working)
# Training phase
step.train()
train_stats = train_loader_step(step)
train_acc = train_stats.accuracy() * 100
train_loss = train_stats.loss()

# Validation phase
with torch.no_grad():
    step.eval()
    val_stats = val_loader_step(step)
    val_acc = val_stats.accuracy() * 100
    val_loss = val_stats.loss()
```

## Current Configuration

### Model Architecture
- **Model:** `BalancedDevignModel` (from `balanced_training_config.py`)
- **Input Dimension:** 100 (matches current Word2Vec embedding size)
- **Architecture:** GatedGraphConv + BatchNorm + Dual Pooling + 3-layer MLP
- **Features:** Residual connections, batch normalization, balanced regularization

### Search Space
- **Learning Rate:** log-uniform [1e-5, 1e-2]
- **Weight Decay:** log-uniform [1e-7, 1e-4]
- **Dropout:** uniform [0.1, 0.5]
- **GNN Steps:** integer [2, 6]
- **Hidden Dimension:** categorical [128, 200, 256, 384]
- **Batch Size:** categorical [4, 8, 16, 32]

### Data Pipeline
- **Input Path:** Uses `configs.Paths().input` (correct path)
- **Data Loading:** Uses `data.loads()` and `data.train_val_test_split()`
- **Dataset Size:** 7,384 samples total
- **Split:** Automatic train/val/test split with shuffling

## Dependencies Added
- **scikit-optimize:** For Bayesian optimization (`pip install scikit-optimize`)

## Verification
✅ All imports work correctly
✅ Model creation successful with current architecture
✅ Data loading works with existing input files
✅ Search space properly defined
✅ Training loop compatible with current Step-based approach

## Usage
```bash
# Run Bayesian hyperparameter optimization
python auto_hyperparameter_bayesian.py

# Test the fixes
python test_bayesian_hyperopt.py
```

## Results
The updated script will:
1. Load 7,384 samples from the input dataset
2. Run 30 Bayesian optimization trials (configurable)
3. Test different hyperparameter combinations intelligently
4. Save results to `hyperparameter_results_bayesian.json`
5. Provide early stopping for stuck models
6. Track best configurations and performance metrics

The Bayesian optimization is now fully compatible with your current model architecture and data pipeline.