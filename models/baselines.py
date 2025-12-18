"""
Baseline GNN Models: GCN and GAT

These models serve as baselines to compare against the graph transformers (GOAT, Exphormer).

Models:
- GCN (Graph Convolutional Network): Kipf & Welling, ICLR 2017
- GAT (Graph Attention Network): Veličković et al., ICLR 2018

These are simpler models with O(E) complexity for message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from typing import Optional


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN).
    
    Paper: Kipf & Welling "Semi-Supervised Classification with Graph Convolutional Networks" (ICLR 2017)
    
    Key Ideas:
    - Local message passing using normalized adjacency
    - Each layer aggregates 1-hop neighbor information
    - Spectral interpretation: low-pass filtering on graph
    
    Complexity: O(E) per layer where E is the number of edges
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of GCN layers
        dropout: Dropout rate
        task_type: 'graph_classification' or 'node_classification'
        pooling_type: Pooling method for graph-level tasks
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
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
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
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
        
        # Convert to float if needed (ZINC dataset has integer node features)
        if x.dtype != torch.float32:
            x = x.float()
        
        # Apply GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling if needed
        if self.task_type == "graph_classification":
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device)
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


class GAT(nn.Module):
    """
    Graph Attention Network (GAT).
    
    Paper: Veličković et al. "Graph Attention Networks" (ICLR 2018)
    
    Key Ideas:
    - Attention-based message passing
    - Learns importance weights for different neighbors
    - Multi-head attention for stability
    
    Complexity: O(E) per layer, but with higher constant due to attention computation
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        task_type: 'graph_classification' or 'node_classification'
        pooling_type: Pooling method for graph-level tasks
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        task_type: str = "graph_classification",
        pooling_type: str = "mean",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.task_type = task_type
        self.pooling_type = pooling_type
        
        # Ensure hidden_channels is divisible by num_heads
        assert hidden_channels % num_heads == 0, \
            f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
        
        head_dim = hidden_channels // num_heads
        
        # Build GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(in_channels, head_dim, heads=num_heads, dropout=dropout, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, head_dim, heads=num_heads, dropout=dropout, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels, head_dim, heads=num_heads, dropout=dropout, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
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
        
        # Convert to float if needed (ZINC dataset has integer node features)
        if x.dtype != torch.float32:
            x = x.float()
        
        # Apply GAT layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)  # GAT typically uses ELU
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling if needed
        if self.task_type == "graph_classification":
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device)
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


class GraphMLP(nn.Module):
    """
    Simple MLP baseline (no graph structure).
    
    This baseline ignores graph structure entirely - just uses node features.
    Useful to show the benefit of using graph structure.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of MLP layers
        dropout: Dropout rate
        task_type: 'graph_classification' or 'node_classification'
        pooling_type: Pooling method for graph-level tasks
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
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
        
        # Build MLP layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
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
        x = data.x
        
        # Convert to float if needed (ZINC dataset has integer node features)
        if x.dtype != torch.float32:
            x = x.float()
        
        # Apply MLP to node features
        x = self.mlp(x)
        
        # Graph-level pooling if needed
        if self.task_type == "graph_classification":
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device)
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





