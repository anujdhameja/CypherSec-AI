# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch_geometric.nn.conv import GatedGraphConv

# torch.manual_seed(2020)


# def get_conv_mp_out_size(in_size, last_layer, mps):
#     size = in_size

#     for mp in mps:
#         size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

#     size = size + 1 if size % 2 != 0 else size

#     return int(size * last_layer["out_channels"])


# def init_weights(m):
#     if type(m) == nn.Linear or type(m) == nn.Conv1d:
#         torch.nn.init.xavier_uniform_(m.weight)


# class Conv(nn.Module):

#     def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
#         super(Conv, self).__init__()
#         self.conv1d_1_args = conv1d_1
#         self.conv1d_1 = nn.Conv1d(**conv1d_1)
#         self.conv1d_2 = nn.Conv1d(**conv1d_2)

#         fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
#         fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])

#         # Dense layers
#         self.fc1 = nn.Linear(fc1_size, 1)
#         self.fc2 = nn.Linear(fc2_size, 1)

#         # Dropout
#         self.drop = nn.Dropout(p=0.2)

#         self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
#         self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

#     def forward(self, hidden, x):
#         concat = torch.cat([hidden, x], 1)
#         concat_size = hidden.shape[1] + x.shape[1]
#         concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

#         Z = self.mp_1(F.relu(self.conv1d_1(concat)))
#         Z = self.mp_2(self.conv1d_2(Z))

#         hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

#         Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
#         Y = self.mp_2(self.conv1d_2(Y))

#         Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
#         Y_flatten_size = int(Y.shape[1] * Y.shape[-1])

#         Z = Z.view(-1, Z_flatten_size)
#         Y = Y.view(-1, Y_flatten_size)
#         res = self.fc1(Z) * self.fc2(Y)
#         res = self.drop(res)
#         # res = res.mean(1)
#         # print(res, mean)
#         sig = torch.sigmoid(torch.flatten(res))
#         return sig


# class Net(nn.Module):

#     def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
#         super(Net, self).__init__()
#         self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
#         self.conv = Conv(**conv_args,
#                          fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
#                          fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
#         # self.conv.apply(init_weights)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.ggc(x, edge_index)
#         x = self.conv(x, data.x)

#         return x

#     def save(self, path):
#         torch.save(self.state_dict(), path)

#     def load(self, path):
#         self.load_state_dict(torch.load(path))



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GatedGraphConv, global_max_pool


# class DevignModel(nn.Module):
#     """
#     Fixed Devign model with input projection layer
    
#     Architecture:
#     1. Input Projection: nodes_dim → hidden_dim
#     2. GatedGraphConv layers
#     3. Conv1d temporal layers
#     4. Global pooling
#     5. Classification head
#     """
    
#     def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8, 
#                  hidden_dim=200, conv1d_channels=[200, 200], dropout=0.3):
#         """
#         Args:
#             input_dim: Node feature dimension (e.g., 205 from Word2Vec)
#             output_dim: Number of classes (e.g., 2 for binary classification)
#             max_edge_types: Number of edge types (usually 1 for single edge type)
#             num_steps: Number of GatedGraphConv propagation steps
#             hidden_dim: Hidden dimension for GNN (must be >= input_dim for GatedGraphConv)
#             conv1d_channels: List of Conv1d output channels
#             dropout: Dropout probability
#         """
#         super(DevignModel, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.max_edge_types = max_edge_types
#         self.num_steps = num_steps
#         self.hidden_dim = hidden_dim
#         self.dropout = dropout
        
#         print(f"\n=== Devign Model Configuration ===")
#         print(f"Input dim: {input_dim}")
#         print(f"Hidden dim: {hidden_dim}")
#         print(f"Output dim: {output_dim}")
#         print(f"GNN steps: {num_steps}")
#         print(f"Conv1d channels: {conv1d_channels}")
#         print(f"Dropout: {dropout}")
        
#         # CRITICAL FIX: Input projection layer
#         # GatedGraphConv requires out_channels >= in_channels
#         # So we project input_dim -> hidden_dim first
#         if input_dim != hidden_dim:
#             self.input_projection = nn.Linear(input_dim, hidden_dim)
#             print(f"✓ Input projection: {input_dim} → {hidden_dim}")
#         else:
#             self.input_projection = None
#             print(f"✓ No projection needed (input_dim == hidden_dim)")
        
#         # GatedGraphConv layer
#         self.ggc = GatedGraphConv(
#             out_channels=hidden_dim,
#             num_layers=num_steps,
#             aggr='add'
#         )
#         print(f"✓ GatedGraphConv: {hidden_dim} channels, {num_steps} steps")
        
#         # Conv1d layers for temporal processing
#         self.conv1d_layers = nn.ModuleList()
#         in_channels = hidden_dim
#         for out_channels in conv1d_channels:
#             self.conv1d_layers.append(
#                 nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
#             )
#             in_channels = out_channels
#         print(f"✓ Conv1d layers: {len(conv1d_channels)} layers")
        
#         # Dropout
#         self.dropout_layer = nn.Dropout(dropout)
        
#         # Classification head
#         final_dim = conv1d_channels[-1] if conv1d_channels else hidden_dim
#         self.fc1 = nn.Linear(final_dim, final_dim // 2)
#         self.fc2 = nn.Linear(final_dim // 2, output_dim)
#         print(f"✓ Classifier: {final_dim} → {final_dim//2} → {output_dim}")
#         print(f"{'='*50}\n")
    
#     def forward(self, data):
#         """
#         Forward pass
        
#         Args:
#             data: PyTorch Geometric Data object with:
#                 - x: Node features [num_nodes, input_dim]
#                 - edge_index: Edge connectivity [2, num_edges]
#                 - batch: Batch assignment vector [num_nodes]
        
#         Returns:
#             out: Predictions [batch_size, output_dim]
#         """
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         # Debug shapes
#         # print(f"Input x: {x.shape}, edge_index: {edge_index.shape}")
        
#         # Step 1: Project input features if needed
#         if self.input_projection is not None:
#             x = self.input_projection(x)
#             x = F.relu(x)
#             # print(f"After projection: {x.shape}")
        
#         # Step 2: GatedGraphConv
#         x = self.ggc(x, edge_index)
#         x = F.relu(x)
#         # print(f"After GGC: {x.shape}")
        
#         # Step 3: Global pooling to get graph-level features
#         x = global_max_pool(x, batch)  # [batch_size, hidden_dim]
#         # print(f"After pooling: {x.shape}")
        
#         # Step 4: Conv1d layers (need to add sequence dimension)
#         # Reshape: [batch_size, hidden_dim] → [batch_size, hidden_dim, 1]
#         x = x.unsqueeze(-1)
        
#         for conv_layer in self.conv1d_layers:
#             x = conv_layer(x)
#             x = F.relu(x)
#             x = self.dropout_layer(x)
        
#         # Remove sequence dimension: [batch_size, channels, 1] → [batch_size, channels]
#         x = x.squeeze(-1)
#         # print(f"After Conv1d: {x.shape}")
        
#         # Step 5: Classification head
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout_layer(x)
        
#         x = self.fc2(x)
#         # print(f"Output: {x.shape}")
        
#         return x


# # ============================================
# # Alternative: Simpler Model (if above fails)
# # ============================================

# class SimpleDevignModel(nn.Module):
#     """
#     Simplified version without Conv1d layers
#     """
    
#     def __init__(self, input_dim, output_dim, hidden_dim=200, num_steps=8, dropout=0.3):
#         super(SimpleDevignModel, self).__init__()
        
#         print(f"\n=== Simple Devign Model ===")
#         print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        
#         # Input projection
#         self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        
#         # GNN
#         self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps)
        
#         # Classifier
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         # Project input
#         if self.input_projection is not None:
#             x = F.relu(self.input_projection(x))
        
#         # GNN
#         x = F.relu(self.ggc(x, edge_index))
        
#         # Pool
#         x = global_max_pool(x, batch)
        
#         # Classify
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
        
#         return x


# # ============================================
# # Model Factory
# # ============================================

# def create_devign_model(input_dim, output_dim=2, model_type='full', **kwargs):
#     """
#     Factory function to create Devign models
    
#     Args:
#         input_dim: Node feature dimension (e.g., 205)
#         output_dim: Number of classes (default: 2)
#         model_type: 'full' or 'simple'
#         **kwargs: Additional model parameters
    
#     Returns:
#         model: Devign model instance
#     """
#     if model_type == 'simple':
#         return SimpleDevignModel(
#             input_dim=input_dim,
#             output_dim=output_dim,
#             hidden_dim=kwargs.get('hidden_dim', 200),
#             num_steps=kwargs.get('num_steps', 8),
#             dropout=kwargs.get('dropout', 0.3)
#         )
#     else:
#         return DevignModel(
#             input_dim=input_dim,
#             output_dim=output_dim,
#             max_edge_types=kwargs.get('max_edge_types', 1),
#             num_steps=kwargs.get('num_steps', 8),
#             hidden_dim=kwargs.get('hidden_dim', 200),
#             conv1d_channels=kwargs.get('conv1d_channels', [200, 200]),
#             dropout=kwargs.get('dropout', 0.3)
#         )


# # ============================================
# # Usage Example
# # ============================================

# if __name__ == "__main__":
#     # Example: Create model for your data
#     model = create_devign_model(
#         input_dim=205,      # Your Word2Vec dimension
#         output_dim=2,       # Binary classification
#         model_type='full',  # or 'simple'
#         hidden_dim=200,     # GNN hidden dimension
#         num_steps=8,        # GNN propagation steps
#         conv1d_channels=[200, 200],
#         dropout=0.3
#     )
    
#     print(model)
    
#     # Test with dummy data
#     from torch_geometric.data import Data, Batch
    
#     # Create dummy graph
#     x = torch.randn(10, 205)  # 10 nodes, 205 features
#     edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
#     data = Data(x=x, edge_index=edge_index)
    
#     # Create batch
#     batch_data = Batch.from_data_list([data, data])
    
#     # Forward pass
#     model.eval()
#     with torch.no_grad():
#         output = model(batch_data)
#         print(f"\nTest output shape: {output.shape}")  # Should be [2, 2]






import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_max_pool


class DevignModel(nn.Module):
    """
    Fixed Devign model with input projection layer
    
    Architecture:
    1. Input Projection: nodes_dim → hidden_dim
    2. GatedGraphConv layers
    3. Conv1d temporal layers
    4. Global pooling
    5. Classification head
    """
    
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8, 
                 hidden_dim=200, conv1d_channels=[200, 200], dropout=0.3):
        """
        Args:
            input_dim: Node feature dimension (e.g., 205 from Word2Vec)
            output_dim: Number of classes (e.g., 2 for binary classification)
            max_edge_types: Number of edge types (usually 1 for single edge type)
            num_steps: Number of GatedGraphConv propagation steps
            hidden_dim: Hidden dimension for GNN (must be >= input_dim for GatedGraphConv)
            conv1d_channels: List of Conv1d output channels
            dropout: Dropout probability
        """
        super(DevignModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.path = None  # For save functionality
        
        print(f"\n=== Devign Model Configuration ===")
        print(f"Input dim: {input_dim}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Output dim: {output_dim}")
        print(f"GNN steps: {num_steps}")
        print(f"Conv1d channels: {conv1d_channels}")
        print(f"Dropout: {dropout}")
        
        # CRITICAL FIX: Input projection layer
        # GatedGraphConv requires out_channels >= in_channels
        # So we project input_dim -> hidden_dim first
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
            print(f"✓ Input projection: {input_dim} → {hidden_dim}")
        else:
            self.input_projection = None
            print(f"✓ No projection needed (input_dim == hidden_dim)")
        
        # GatedGraphConv layer
        self.ggc = GatedGraphConv(
            out_channels=hidden_dim,
            num_layers=num_steps,
            aggr='add'
        )
        print(f"✓ GatedGraphConv: {hidden_dim} channels, {num_steps} steps")
        
        # Conv1d layers for temporal processing
        self.conv1d_layers = nn.ModuleList()
        in_channels = hidden_dim
        for out_channels in conv1d_channels:
            self.conv1d_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            in_channels = out_channels
        print(f"✓ Conv1d layers: {len(conv1d_channels)} layers")
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Classification head
        final_dim = conv1d_channels[-1] if conv1d_channels else hidden_dim
        self.fc1 = nn.Linear(final_dim, final_dim // 2)
        self.fc2 = nn.Linear(final_dim // 2, output_dim)
        print(f"✓ Classifier: {final_dim} → {final_dim//2} → {output_dim}")
        print(f"{'='*50}\n")
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch assignment vector [num_nodes]
        
        Returns:
            out: Predictions [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Debug shapes
        # print(f"Input x: {x.shape}, edge_index: {edge_index.shape}")
        
        # Step 1: Project input features if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
            x = F.relu(x)
            # print(f"After projection: {x.shape}")
        
        # Step 2: GatedGraphConv
        x = self.ggc(x, edge_index)
        x = F.relu(x)
        # print(f"After GGC: {x.shape}")
        
        # Step 3: Global pooling to get graph-level features
        x = global_max_pool(x, batch)  # [batch_size, hidden_dim]
        # print(f"After pooling: {x.shape}")
        
        # Step 4: Conv1d layers (need to add sequence dimension)
        # Reshape: [batch_size, hidden_dim] → [batch_size, hidden_dim, 1]
        x = x.unsqueeze(-1)
        
        for conv_layer in self.conv1d_layers:
            x = conv_layer(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Remove sequence dimension: [batch_size, channels, 1] → [batch_size, channels]
        x = x.squeeze(-1)
        # print(f"After Conv1d: {x.shape}")
        
        # Step 5: Classification head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        x = self.fc2(x)
        # print(f"Output: {x.shape}")
        
        return x


# ============================================
# Alternative: Simpler Model (if above fails)
# ============================================

class SimpleDevignModel(nn.Module):
    """
    Simplified version without Conv1d layers
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=200, num_steps=8, dropout=0.3):
        super(SimpleDevignModel, self).__init__()
        
        print(f"\n=== Simple Devign Model ===")
        print(f"Input: {input_dim} → Hidden: {hidden_dim} → Output: {output_dim}")
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        
        # GNN
        self.ggc = GatedGraphConv(hidden_dim, num_layers=num_steps)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Project input
        if self.input_projection is not None:
            x = F.relu(self.input_projection(x))
        
        # GNN
        x = F.relu(self.ggc(x, edge_index))
        
        # Pool
        x = global_max_pool(x, batch)
        
        # Classify
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================
# Model Factory
# ============================================

def create_devign_model(input_dim, output_dim=2, model_type='full', **kwargs):
    """
    Factory function to create Devign models
    
    Args:
        input_dim: Node feature dimension (e.g., 205)
        output_dim: Number of classes (default: 2)
        model_type: 'full' or 'simple'
        **kwargs: Additional model parameters
    
    Returns:
        model: Devign model instance
    """
    if model_type == 'simple':
        return SimpleDevignModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=kwargs.get('hidden_dim', 200),
            num_steps=kwargs.get('num_steps', 8),
            dropout=kwargs.get('dropout', 0.3)
        )
    else:
        return DevignModel(
            input_dim=input_dim,
            output_dim=output_dim,
            max_edge_types=kwargs.get('max_edge_types', 1),
            num_steps=kwargs.get('num_steps', 8),
            hidden_dim=kwargs.get('hidden_dim', 200),
            conv1d_channels=kwargs.get('conv1d_channels', [200, 200]),
            dropout=kwargs.get('dropout', 0.3)
        )


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    # Example: Create model for your data
    model = create_devign_model(
        input_dim=205,      # Your Word2Vec dimension
        output_dim=2,       # Binary classification
        model_type='full',  # or 'simple'
        hidden_dim=200,     # GNN hidden dimension
        num_steps=8,        # GNN propagation steps
        conv1d_channels=[200, 200],
        dropout=0.3
    )
    
    print(model)
    
    # Test with dummy data
    from torch_geometric.data import Data, Batch
    
    # Create dummy graph
    x = torch.randn(10, 205)  # 10 nodes, 205 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    
    # Create batch
    batch_data = Batch.from_data_list([data, data])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_data)
        print(f"\nTest output shape: {output.shape}")  # Should be [2, 2]