import os
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

class InputDataset(TorchDataset):
    def __init__(self, data_dir, max_files=None):
        """
        Args:
            data_dir (str): Directory containing .pkl files with data
            max_files (int, optional): Maximum number of files to load (for testing)
        """
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
        
        # Limit number of files if specified (for testing)
        if max_files is not None:
            self.file_list = self.file_list[:max_files]
        
        self.data = []
        self.targets = []
        
        # Load data from files
        self._load_data()
    
    def _load_data(self):
        """Load data from all files into memory."""
        total_samples = 0
        
        for file in self.file_list:
            file_path = os.path.join(self.data_dir, file)
            try:
                # Load the DataFrame
                df = pd.read_pickle(file_path)
                
                # Extract data and targets
                if not df.empty and 'input' in df.columns and 'target' in df.columns:
                    self.data.extend(df['input'].tolist())
                    self.targets.extend(df['target'].tolist())
                    total_samples += len(df)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(self.data)} samples from {len(self.file_list)} files")
        
        # Verify data consistency
        if len(self.data) != len(self.targets):
            raise ValueError(f"Mismatch between number of samples ({len(self.data)}) and targets ({len(self.targets)})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single data sample and its target."""
        data = self.data[idx]
        target = self.targets[idx]
        
        # Ensure the data is a PyG Data object
        if not isinstance(data, Data):
            raise TypeError(f"Expected data to be a PyG Data object, got {type(data)}")
        
        # Add target to the data object if it doesn't have one
        if not hasattr(data, 'y'):
            data.y = torch.tensor([target], dtype=torch.long)
        
        return data
    
    def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
        """Get a DataLoader for this dataset.
        
        Args:
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the data
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            DataLoader: PyTorch Geometric DataLoader for this dataset
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # Important: PyG's collate function handles batching of Data objects
            collate_fn=lambda batch: Batch.from_data_list(batch)
        )
