#!/usr/bin/env python3
"""
Distributed training setup for vulnerability detection model
Supports multi-machine training with PyTorch DDP
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
import argparse
import json
from pathlib import Path

import configs
import src.data as data
import src.process as process
from src.process.model import create_devign_model


def setup_distributed(rank, world_size, master_addr='localhost', master_port='12355'):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size
    )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
    
    return device


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def create_distributed_dataloader(dataset, batch_size, rank, world_size, shuffle=True):
    """Create distributed dataloader"""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,  # Reduce workers to save memory
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader, sampler


def train_distributed(rank, world_size, config, master_addr='localhost', master_port='12355'):
    """Main distributed training function"""
    
    print(f"Starting process {rank}/{world_size}")
    
    # Setup distributed training
    device = setup_distributed(rank, world_size, master_addr, master_port)
    
    try:
        # Load configuration
        PATHS = configs.Paths()
        FILES = configs.Files()
        devign_config = configs.Devign()
        process_config = configs.Process()
        
        # Load dataset
        print(f"Rank {rank}: Loading dataset...")
        input_dataset = data.loads(PATHS.input)
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = data.train_val_test_split(
            input_dataset, 
            shuffle=process_config.shuffle
        )
        
        # Create distributed dataloaders
        train_loader, train_sampler = create_distributed_dataloader(
            train_dataset, 
            process_config.batch_size // world_size,  # Reduce batch size per process
            rank, 
            world_size, 
            shuffle=True
        )
        
        val_loader, val_sampler = create_distributed_dataloader(
            val_dataset,
            process_config.batch_size // world_size,
            rank,
            world_size,
            shuffle=False
        )
        
        # Create model
        print(f"Rank {rank}: Creating model...")
        model = create_devign_model(
            input_dim=205,  # From your config
            output_dim=2,
            model_type='full',
            hidden_dim=devign_config.model['gated_graph_conv_args']['out_channels'],
            num_steps=devign_config.model['gated_graph_conv_args']['num_layers'],
            dropout=0.3
        ).to(device)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[device] if torch.cuda.is_available() else None)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=devign_config.learning_rate,
            weight_decay=devign_config.weight_decay
        )
        
        # Setup loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        print(f"Rank {rank}: Starting training...")
        for epoch in range(process_config.epochs):
            # Set epoch for sampler (important for shuffling)
            train_sampler.set_epoch(epoch)
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), process_config.gradient_clip)
                
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch.y.size(0)
                train_correct += (predicted == batch.y).sum().item()
                
                if batch_idx % 10 == 0 and rank == 0:  # Only print from rank 0
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation phase (only on rank 0 to avoid duplicate evaluation)
            if rank == 0:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        outputs = model(batch)
                        loss = criterion(outputs, batch.y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch.y.size(0)
                        val_correct += (predicted == batch.y).sum().item()
                
                # Print epoch results
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total
                
                print(f'Epoch {epoch+1}/{process_config.epochs}:')
                print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
                
                # Save checkpoint
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = f"{PATHS.model}checkpoint_epoch_{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f'Checkpoint saved: {checkpoint_path}')
        
        # Save final model (only from rank 0)
        if rank == 0:
            final_model_path = f"{PATHS.model}distributed_final_model.pth"
            torch.save(model.module.state_dict(), final_model_path)
            print(f'Final model saved: {final_model_path}')
    
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('--world-size', type=int, default=2, help='Number of processes/machines')
    parser.add_argument('--rank', type=int, default=0, help='Rank of current process')
    parser.add_argument('--master-addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master-port', type=str, default='12355', help='Master node port')
    
    args = parser.parse_args()
    
    # Load config
    with open('configs.json', 'r') as f:
        config = json.load(f)
    
    # Start distributed training
    train_distributed(
        rank=args.rank,
        world_size=args.world_size,
        config=config,
        master_addr=args.master_addr,
        master_port=args.master_port
    )


if __name__ == '__main__':
    main()