"""
Base Graph Transformer class with common functionality.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from typing import Optional


class BaseGraphTransformer(nn.Module):
    """
    Base class for graph transformers.
    
    Provides common functionality:
    - Input projection
    - Positional encoding integration
    - Output heads for different tasks
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        pe_dim: int = 8,
        dropout: float = 0.0,
        task_type: str = "graph_classification",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.pe_dim = pe_dim
        self.dropout = dropout
        self.task_type = task_type
        
        # Input projection: node features + positional encoding
        self.node_encoder = nn.Linear(in_channels + pe_dim, hidden_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output head (will be set by subclasses)
        self.output_head = None
    
    def create_output_head(self, pooling_type: str = "mean"):
        """
        Create output head for different task types.
        
        Args:
            pooling_type: 'mean', 'max', 'add', or 'attention'
        """
        if self.task_type == "graph_classification":
            # Graph-level tasks: need pooling + classifier
            self.pooling_type = pooling_type
            self.output_head = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.out_channels)
            )
        elif self.task_type == "node_classification":
            # Node-level tasks: direct prediction
            self.output_head = nn.Linear(self.hidden_channels, self.out_channels)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Pool node representations to graph representation.
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph-level features [batch_size, hidden_channels]
        """
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
    
    def encode_input(self, data: Data) -> torch.Tensor:
        """
        Encode input node features with positional encodings.
        
        Args:
            data: PyG Data object with x and pe attributes
            
        Returns:
            Encoded node features [num_nodes, hidden_channels]
        """
        # Concatenate node features and positional encodings
        if hasattr(data, 'pe') and data.pe is not None:
            x = torch.cat([data.x, data.pe], dim=-1)
        else:
            # If no PE, pad with zeros
            pe = torch.zeros(data.x.size(0), self.pe_dim, device=data.x.device)
            x = torch.cat([data.x, pe], dim=-1)
        
        # Project to hidden dimension
        x = self.node_encoder(x)
        x = self.dropout_layer(x)
        
        return x
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass (to be implemented by subclasses).
        """
        raise NotImplementedError("Subclasses must implement forward()")


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean pooling over batch"""
    batch_size = batch.max().item() + 1
    return torch.stack([x[batch == i].mean(0) for i in range(batch_size)])


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Max pooling over batch"""
    batch_size = batch.max().item() + 1
    return torch.stack([x[batch == i].max(0)[0] for i in range(batch_size)])


def global_add_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Sum pooling over batch"""
    batch_size = batch.max().item() + 1
    return torch.stack([x[batch == i].sum(0) for i in range(batch_size)])


