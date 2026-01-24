"""
Hybrid GNN Models combining local message passing with global mechanisms.

Models:
- GCNVirtualNode: GCN with Virtual Node for global information exchange
- GATVirtualNode: GAT with Virtual Node

The Virtual Node approach is used in OGB baselines and provides a simple way
to add global context to message passing GNNs without full attention complexity.

Reference: 
- OGB Paper: Hu et al. "Open Graph Benchmark" (NeurIPS 2020)
- Virtual Node is also used in GOAT as a form of global attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_scatter import scatter
from typing import Optional


class VirtualNodeLayer(nn.Module):
    """
    Virtual Node mechanism for global information aggregation.
    
    The virtual node is connected to all nodes in the graph:
    1. Aggregate all node features to the virtual node
    2. Transform the virtual node
    3. Broadcast virtual node info back to all nodes
    
    This enables O(N) global communication instead of O(N²) full attention.
    """
    
    def __init__(self, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
        
        # MLP to transform virtual node
        self.virtual_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        x: torch.Tensor,
        virtual_node: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple:
        """
        Update nodes with virtual node information.
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            virtual_node: Virtual node features [batch_size, hidden_channels]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Updated (x, virtual_node) tuple
        """
        batch_size = virtual_node.size(0)
        
        # 1. Add virtual node info to all nodes (broadcast)
        virtual_to_nodes = virtual_node[batch]  # [num_nodes, hidden_channels]
        x = x + virtual_to_nodes
        
        # 2. Aggregate node features to update virtual node
        # Global mean pooling per graph
        node_to_virtual = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # 3. Update virtual node with residual connection
        virtual_node_updated = self.virtual_mlp(node_to_virtual)
        alpha = torch.sigmoid(self.residual_weight)
        virtual_node = alpha * virtual_node + (1 - alpha) * virtual_node_updated
        
        return x, virtual_node


class GCNVirtualNode(nn.Module):
    """
    GCN with Virtual Node for global information exchange.
    
    Combines local message passing (GCN) with global aggregation via a virtual node.
    This is a hybrid approach between pure GNNs and Transformers.
    
    Key Ideas:
    - Each GCN layer is followed by virtual node update
    - Virtual node aggregates global graph information
    - Broadcasts global info back to all nodes
    - O(E + N) complexity instead of O(N²) for full attention
    
    This approach is used in OGB baselines and achieves strong performance.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of GCN + VirtualNode blocks
        dropout: Dropout rate
        task_type: 'graph_classification' or 'node_classification'
        pooling_type: Pooling method for graph-level tasks
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        task_type: str = "graph_classification",
        pooling_type: str = "mean",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.task_type = task_type
        self.pooling_type = pooling_type
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Virtual node embedding (learnable initial state)
        self.virtual_node_embedding = nn.Embedding(1, hidden_channels)
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Virtual node layers (one after each GCN layer)
        self.virtual_node_layers = nn.ModuleList([
            VirtualNodeLayer(hidden_channels, dropout)
            for _ in range(num_layers - 1)  # No VN update after last layer
        ])
        
        # Output head
        if task_type == "graph_classification":
            self.output_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels)
            )
        else:
            self.output_head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass."""
        x, edge_index = data.x, data.edge_index
        
        # Convert to float if needed
        if x.dtype != torch.float32:
            x = x.float()
        
        # Get batch assignment
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        batch_size = batch.max().item() + 1
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        # Initialize virtual node for each graph in batch
        virtual_node = self.virtual_node_embedding.weight.expand(batch_size, -1)
        
        # Apply GCN + Virtual Node layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # GCN layer
            x_conv = conv(x, edge_index)
            x_conv = bn(x_conv)
            x_conv = F.relu(x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            
            # Residual connection for GCN
            x = x + x_conv
            
            # Virtual node update (except for last layer)
            if i < len(self.virtual_node_layers):
                x, virtual_node = self.virtual_node_layers[i](x, virtual_node, batch)
        
        # Graph-level pooling if needed
        if self.task_type == "graph_classification":
            x = self._pool_graph(x, batch)
        
        # Output
        out = self.output_head(x)
        return out
    
    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node features to graph-level representation."""
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class GINVirtualNode(nn.Module):
    """
    GIN with Virtual Node - combining the expressiveness of GIN with global context.
    
    This is a stronger hybrid model that uses:
    - GIN layers for expressive local message passing
    - Virtual node for global information exchange
    - Jumping Knowledge to use all layer representations
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of GIN + VirtualNode blocks
        dropout: Dropout rate
        train_eps: If True, learn epsilon parameter in GIN
        task_type: 'graph_classification' or 'node_classification'
        pooling_type: Pooling method for graph-level tasks
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        train_eps: bool = True,
        task_type: str = "graph_classification",
        pooling_type: str = "add",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.task_type = task_type
        self.pooling_type = pooling_type
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Virtual node embedding
        self.virtual_node_embedding = nn.Embedding(1, hidden_channels)
        
        # GIN layers with MLPs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Virtual node layers
        self.virtual_node_layers = nn.ModuleList([
            VirtualNodeLayer(hidden_channels, dropout)
            for _ in range(num_layers - 1)
        ])
        
        # Output head with Jumping Knowledge
        if task_type == "graph_classification":
            self.output_head = nn.Sequential(
                nn.Linear(hidden_channels * num_layers, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels)
            )
        else:
            self.output_head = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass."""
        x, edge_index = data.x, data.edge_index
        
        # Convert to float if needed
        if x.dtype != torch.float32:
            x = x.float()
        
        # Get batch assignment
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        batch_size = batch.max().item() + 1
        
        # Project input
        x = self.input_proj(x)
        
        # Initialize virtual node
        virtual_node = self.virtual_node_embedding.weight.expand(batch_size, -1)
        
        # Store layer outputs for jumping knowledge
        layer_outputs = []
        
        # Apply GIN + Virtual Node layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # GIN layer
            x_conv = conv(x, edge_index)
            x_conv = bn(x_conv)
            x_conv = F.relu(x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            
            # Residual connection
            x = x + x_conv
            
            # Virtual node update
            if i < len(self.virtual_node_layers):
                x, virtual_node = self.virtual_node_layers[i](x, virtual_node, batch)
            
            layer_outputs.append(x)
        
        # Graph-level pooling with Jumping Knowledge
        if self.task_type == "graph_classification":
            pooled_outputs = []
            for layer_out in layer_outputs:
                pooled_outputs.append(self._pool_graph(layer_out, batch))
            x = torch.cat(pooled_outputs, dim=-1)
        
        # Output
        out = self.output_head(x)
        return out
    
    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool node features to graph-level representation."""
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
