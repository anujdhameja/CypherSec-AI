"""
CORRECTED: Load the ACTUAL Config 9 model architecture
Based on the real saved model parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from pathlib import Path
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.objects.input_dataset import InputDataset


class ActualConfig9Model(nn.Module):
    """
    The ACTUAL Config 9 model architecture based on saved weights
    - hidden_dim: 384 (not 256!)
    - num_steps: 4 (not 5!)
    - Uses attention pooling (not mean_max!)
    """
    
    def __init__(self, input_dim=100, output_dim=2, hidden_dim=384, 
                 num_steps=4, dropout=0.2):
        super().__init__()
        
        print(f"\n=== ACTUAL Config 9 Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        print(f"✓ Real Config 9 parameters")
        print(f"   - GNN steps: {num_steps}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Pooling: attention")
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # GNN
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps, aggr='add')
        self.gnn_bn = nn.BatchNorm1d(hidden_dim)
        
        # Attention pooling (this is what the saved model actually uses!)
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Classifier (single pooling, not dual!)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        print("="*50 + "\n")
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GNN with residual
        x_skip = x
        x = self.ggc(x, edge_index)
        x = self.gnn_bn(x)
        x = F.relu(x + x_skip)  # Residual
        x = self.dropout(x)
        
        # Attention pooling (this is the key difference!)
        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = torch.sum(attention_weights * x, dim=0, keepdim=True)
        
        # Handle batch dimension properly
        batch_size = batch.max().item() + 1
        x_pooled = torch.zeros(batch_size, x.size(1), device=x.device)
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                node_features = x[mask]
                attn_weights = torch.softmax(self.attention(node_features), dim=0)
                x_pooled[i] = torch.sum(attn_weights * node_features, dim=0)
        
        x = x_pooled
        
        # Classifier
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"✓ Model loaded from {path}")


class ActualConfig9Trainer:
    """Train with the ACTUAL Config 9 architecture"""
    
    def __init__(self, pretrained_path='models/production_model_config9_v1.0.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_path = pretrained_path
        
        print("="*80)
        print("ACTUAL CONFIG 9 TRAINER")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Pretrained model: {pretrained_path}")

    def load_pretrained_model(self):
        """Load model with the CORRECT architecture"""
        
        # Create model with ACTUAL saved parameters
        model = ActualConfig9Model(
            input_dim=100,
            output_dim=2,
            hidden_dim=384,  # ACTUAL saved value
            num_steps=4,     # ACTUAL saved value
            dropout=0.2
        )
        
        # Load pretrained weights
        print(f"\n📦 Loading pretrained weights...")
        state_dict = torch.load(self.pretrained_path, map_location=self.device)
        
        # Load weights (should match perfectly now)
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Pretrained weights loaded successfully (strict=True)")
        
        model = model.to(self.device)
        return model

    def get_finetuning_optimizer(self, model):
        """Create optimizer for fine-tuning"""
        finetune_lr = 1e-5
        finetune_weight_decay = 1e-5
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=finetune_lr,
            weight_decay=finetune_weight_decay
        )
        
        print(f"\n⚙️  Fine-tuning Optimizer:")
        print(f"  Learning Rate: {finetune_lr}")
        print(f"  Weight Decay: {finetune_weight_decay}")
        
        return optimizer

    def train_epoch(self, model, loader, optimizer, criterion):
        """Train one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward
            output = model(batch)
            loss = criterion(output, batch.y.long())
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == batch.y.long()).sum().item()
            total += batch.y.size(0)
        
        return total_loss / len(loader), correct / total

    def eval_epoch(self, model, loader, criterion):
        """Evaluate one epoch"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                output = model(batch)
                loss = criterion(output, batch.y.long())
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == batch.y.long()).sum().item()
                total += batch.y.size(0)
        
        return total_loss / len(loader), correct / total

    def train(self, num_epochs=50, batch_size=32):
        """Fine-tune the actual Config 9 model"""
        # Load model
        model = self.load_pretrained_model()
        
        # Load data
        print(f"\n📊 Loading dataset...")
        dataset = InputDataset('data/input')
        
        # Split data
        from sklearn.model_selection import train_test_split
        indices = list(range(len(dataset)))
        labels = [int(dataset[i].y.item()) for i in indices]
        
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=42, stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, 
            stratify=[labels[i] for i in temp_idx]
        )
        
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in val_idx]
        test_dataset = [dataset[i] for i in test_idx]
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss
        optimizer = self.get_finetuning_optimizer(model)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"\n🚀 Starting fine-tuning for {num_epochs} epochs...")
        print("="*80)
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc = self.eval_epoch(model, val_loader, criterion)
            
            print(f"Epoch {epoch+1:2d} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'models/actual_config9_finetuned.pth')
                print(f"  ✅ Best model saved")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⏸️  Early stopping at epoch {epoch+1}")
                    break
        
        # Test evaluation
        print("\n" + "="*80)
        print("FINAL TEST EVALUATION")
        print("="*80)
        
        model.load_state_dict(torch.load('models/actual_config9_finetuned.pth'))
        test_loss, test_acc = self.eval_epoch(model, test_loader, criterion)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Expected: 83.57%")
        print(f"  Difference: {test_acc - 0.8357:+.4f}")
        
        if test_acc > 0.75:
            print(f"\n✅ SUCCESS! Model is working properly!")
        else:
            print(f"\n⚠️  Lower than expected. Check training process.")


if __name__ == '__main__':
    trainer = ActualConfig9Trainer()
    trainer.train(num_epochs=50, batch_size=16)  # Smaller batch for attention