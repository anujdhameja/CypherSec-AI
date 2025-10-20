# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch_geometric.nn import GatedGraphConv, global_max_pool, global_mean_pool

# from ..utils import log
# from .step import Step
# from .balanced_training_config import BalancedDevignModel


# class Devign(Step):
#     def __init__(self,
#                  path: str,
#                  device: str,
#                  model: dict,
#                  learning_rate: float,
#                  weight_decay: float,
#                  loss_lambda: float):
#         self.path = path
#         self.lr = learning_rate * 3  # Increase LR for stability (1e-4 → 3e-4)
#         self.wd = weight_decay
#         self.ll = loss_lambda
#         self.device = device
        
#         log.log_info('devign', f"🔧 BALANCED CONFIG - LR: {self.lr}; WD: {self.wd}; LL: {self.ll};")
        
#         # Create BALANCED model from balanced_training_config
#         _model = BalancedDevignModel(
#             input_dim=model['conv_args']['conv1d_1']['in_channels'],
#             output_dim=2,
#             hidden_dim=model['gated_graph_conv_args']['out_channels'],
#             num_steps=4,  # Using balanced number of steps
#             dropout=0.4   # Balanced dropout for better generalization
#         )
#         _model = _model.to(device)
        
#         # Use CrossEntropy with label smoothing for better regularization
#         class LabelSmoothingLoss(nn.Module):
#             def __init__(self, classes=2, smoothing=0.1):
#                 super(LabelSmoothingLoss, self).__init__()
#                 self.confidence = 1.0 - smoothing
#                 self.smoothing = smoothing
#                 self.classes = classes
            
#             def forward(self, pred, target):
#                 pred = pred.log_softmax(dim=-1)
#                 with torch.no_grad():
#                     true_dist = torch.zeros_like(pred)
#                     true_dist.fill_(self.smoothing / (self.classes - 1))
#                     true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
#                 return torch.mean(torch.sum(-true_dist * pred, dim=-1))
        
#         # Use label smoothing loss for better generalization
#         loss_fn = LabelSmoothingLoss(classes=2, smoothing=0.1)
        
#         # Create optimizer with balanced weight decay
#         optimizer = optim.AdamW(_model.parameters(), lr=self.lr, weight_decay=1e-5, amsgrad=True)
        
#         # Add learning rate scheduler with balanced patience
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer,
#             mode='min',
#             factor=0.5,
#             patience=7  # Slightly more patience for balanced training
#         )
        
#         super().__init__(model=_model,
#                          loss_function=lambda o, t: loss_fn(o, t.squeeze().long()),
#                          optimizer=optimizer)

#         self.count_parameters()

#     def load(self):
#         self.model.load(self.path)

#     def save(self):
#         self.model.save(self.path)

#     def count_parameters(self):
#         count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         print(f"The model has {count:,} trainable parameters")





"""
FIXED Devign Configuration
Based on diagnostic results - removes broken label smoothing
"""

import torch
import torch.nn as nn
import torch.optim as optim

from ..utils import log
from .step import Step
from .balanced_training_config import BalancedDevignModel


class Devign(Step):
    def __init__(self,
                 path: str,
                 device: str,
                 model: dict,
                 learning_rate: float,
                 weight_decay: float,
                 loss_lambda: float):
        self.path = path
        self.lr = learning_rate  # Use original LR (don't multiply by 3!)
        self.wd = weight_decay
        self.ll = loss_lambda
        self.device = device
        
        log.log_info('devign', f"🔧 FIXED CONFIG - LR: {self.lr}; WD: {self.wd};")
        
        # Create model with REDUCED regularization
        # _model = BalancedDevignModel(
        #     input_dim=model['conv_args']['conv1d_1']['in_channels'],
        #     output_dim=2,
        #     hidden_dim=model['gated_graph_conv_args']['out_channels'],
        #     num_steps=4,
        #     dropout=0.2  # REDUCED from 0.4 (diagnostic recommendation)
        # )
        _model = BalancedDevignModel(
            input_dim=100,  # CHANGED from model['conv_args']['conv1d_1']['in_channels']
            output_dim=2,
            hidden_dim=200,
            num_steps=4,
            dropout=0.2
            )
        _model = _model.to(device)
        
        # CRITICAL FIX: Use standard CrossEntropyLoss (no label smoothing)
        loss_fn = nn.CrossEntropyLoss()
        
        log.log_info('devign', "✓ Using standard CrossEntropyLoss (removed label smoothing)")
        
        # Create optimizer with REDUCED weight decay
        optimizer = optim.Adam(  # Use Adam (not AdamW)
            _model.parameters(),
            lr=self.lr,
            weight_decay=1e-6  # REDUCED from 1e-5 (diagnostic recommendation)
        )
        
        log.log_info('devign', f"✓ Reduced regularization: dropout=0.2, wd=1e-6")
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        super().__init__(
            model=_model,
            loss_function=loss_fn,  # Just pass the loss function directly
            optimizer=optimizer
        )

        self.count_parameters()

    def load(self):
        self.model.load(self.path)

    def save(self):
        self.model.save(self.path)

    def count_parameters(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.log_info('devign', f"Model has {count:,} trainable parameters")


# ============================================
# Alternative: Even Simpler Configuration
# ============================================

class DevignMinimal(Step):
    """
    Absolute minimal configuration to get SOMETHING working
    Use this if the above still doesn't work
    """
    
    def __init__(self,
                 path: str,
                 device: str,
                 model: dict,
                 learning_rate: float,
                 weight_decay: float,
                 loss_lambda: float):
        self.path = path
        self.lr = learning_rate * 5  # Higher LR: 1e-4 → 5e-4
        self.wd = 0  # NO weight decay
        self.ll = loss_lambda
        self.device = device
        
        log.log_info('devign', f"🔧 MINIMAL CONFIG - LR: {self.lr}; NO REGULARIZATION")
        
        # Simplest possible model
        _model = BalancedDevignModel(
            input_dim=model['conv_args']['conv1d_1']['in_channels'],
            output_dim=2,
            hidden_dim=model['gated_graph_conv_args']['out_channels'],
            num_steps=3,  # Even fewer steps
            dropout=0.1   # Minimal dropout
        )
        _model = _model.to(device)
        
        # Simple loss and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(_model.parameters(), lr=self.lr, momentum=0.9)
        
        super().__init__(
            model=_model,
            loss_function=loss_fn,
            optimizer=optimizer
        )
        
        self.count_parameters()

    def load(self):
        self.model.load(self.path)

    def save(self):
        self.model.save(self.path)

    def count_parameters(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.log_info('devign', f"Model has {count:,} trainable parameters")