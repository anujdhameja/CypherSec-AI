#!/usr/bin/env python3
"""
Memory-efficient training for single machine with limited RAM
Implements gradient accumulation, mixed precision, and data streaming
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
import gc
import psutil
import os
from pathlib import Path

import configs
import src.data as data
from src.process.model import create_devign_model


class MemoryEfficientTrainer:
    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Mixed precision training
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.use_amp = torch.cuda.is_available()
        
        # Memory monitoring
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self):
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / 1024 / 1024 / 1024
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def train_epoch_with_accumulation(self, dataloader, accumulation_steps=4):
        """Train one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(batch)
                    loss = F.cross_entropy(outputs, batch.y)
                    loss = loss / accumulation_steps  # Scale loss
            else:
                outputs = self.model(batch)
                loss = F.cross_entropy(outputs, batch.y)
                loss = loss / accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch.y.size(0)
            total_correct += (predicted == batch.y).sum().item()
            
            # Memory management
            if batch_idx % 50 == 0:
                memory_gb = self.get_memory_usage()
                print(f'Batch {batch_idx}, Loss: {loss.item()*accumulation_steps:.4f}, Memory: {memory_gb:.2f}GB')
                
                if memory_gb > 14:  # Clear cache if approaching 16GB limit
                    self.clear_cache()
            
            # Clear batch from memory
            del batch, outputs, loss
            
        # Handle remaining gradients
        if len(dataloader) % accumulation_steps != 0:
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validation with memory efficiency"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch)
                        loss = F.cross_entropy(outputs, batch.y)
                else:
                    outputs = self.model(batch)
                    loss = F.cross_entropy(outputs, batch.y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch.y.size(0)
                total_correct += (predicted == batch.y).sum().item()
                
                # Clear batch from memory
                del batch, outputs, loss
                
                if batch_idx % 100 == 0:
                    self.clear_cache()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * total_correct / total_samples
        
        return avg_loss, accuracy


def create_memory_efficient_dataloaders(dataset, batch_size=8, num_workers=2):
    """Create memory-efficient dataloaders with smaller batch sizes"""
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = data.train_val_test_split(
        dataset, shuffle=True
    )
    
    # Create dataloaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to save RAM
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main memory-efficient training function"""
    
    # Configuration
    PATHS = configs.Paths()
    FILES = configs.Files()
    devign_config = configs.Devign()
    process_config = configs.Process()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Memory monitoring
    process = psutil.Process(os.getpid())
    print(f"Initial memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f}GB")
    
    # Load dataset in chunks to save memory
    print("Loading dataset...")
    input_dataset = data.loads(PATHS.input)
    print(f"Dataset size: {len(input_dataset)} samples")
    
    # Create memory-efficient dataloaders
    train_loader, val_loader, test_loader = create_memory_efficient_dataloaders(
        input_dataset, 
        batch_size=8,  # Reduced batch size
        num_workers=1   # Reduced workers
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_devign_model(
        input_dim=205,
        output_dim=2,
        model_type='simple',  # Use simpler model to save memory
        hidden_dim=128,       # Reduced hidden dimension
        num_steps=4,          # Fewer GNN steps
        dropout=0.3
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer with lower learning rate for stability
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=devign_config.learning_rate * 0.5,  # Reduced learning rate
        weight_decay=devign_config.weight_decay
    )
    
    # Setup trainer
    trainer = MemoryEfficientTrainer(model, optimizer, device, process_config)
    
    # Training loop
    print("Starting memory-efficient training...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(process_config.epochs):
        print(f"\nEpoch {epoch+1}/{process_config.epochs}")
        
        # Training
        train_loss, train_acc = trainer.train_epoch_with_accumulation(
            train_loader, 
            accumulation_steps=8  # Accumulate gradients over 8 mini-batches
        )
        
        # Validation
        val_loss, val_acc = trainer.validate(val_loader)
        
        # Print results
        memory_gb = trainer.get_memory_usage()
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Memory usage: {memory_gb:.2f}GB")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), f"{PATHS.model}best_memory_efficient_model.pth")
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= process_config.patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Clear cache after each epoch
        trainer.clear_cache()
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()