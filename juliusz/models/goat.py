"""
GOAT: A Global Transformer on Large-scale Graphs

Paper: Kong et al. "GOAT: A Global Transformer on Large-scale Graphs" (ICML 2023)
Link: https://proceedings.mlr.press/v202/kong23a.html

Key Ideas:
1. Uses virtual "super nodes" to approximate global attention
2. Achieves O(N) complexity instead of O(N²) for full attention
3. Local message passing + global aggregation via super nodes

Architecture:
- Local GNN layers (message passing)
- Global pooling to super nodes
- Super node attention
- Broadcasting back to nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional

from .base import BaseGraphTransformer, global_mean_pool
from .layers import MultiHeadAttention, FeedForward


class GOATLayer(nn.Module):
    """
    Single GOAT layer with local + global attention.
    
    Steps:
    1. Local message passing (GNN-style)
    2. Pool nodes to virtual super nodes
    3. Attend over super nodes (global information)
    4. Broadcast super node info back to nodes
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 8,
        num_virtual_nodes: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_virtual_nodes = num_virtual_nodes
        
        # Local message passing (like GNN)
        self.local_msg = nn.Linear(hidden_channels, hidden_channels)
        
        # Global attention over virtual nodes
        self.global_attention = MultiHeadAttention(
            hidden_channels,
            num_heads,
            dropout
        )
        
        # Virtual node projection
        self.virtual_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        
        # Feed-forward
        self.ffn = FeedForward(hidden_channels, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Updated node features [num_nodes, hidden_channels]
        """
        # 1. Local message passing
        x_local = self._local_message_passing(x, edge_index)
        x = x + self.dropout(x_local)
        x = self.norm1(x)
        
        # 2. Global attention via virtual nodes
        x_global = self._global_attention(x, batch)
        x = x + self.dropout(x_global)
        x = self.norm2(x)
        
        # 3. Feed-forward
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm3(x)
        
        return x
    
    def _local_message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Local message passing along edges (like GNN).
        
        Aggregates neighbor information for each node.
        """
        src, dst = edge_index
        
        # Transform features
        x_msg = self.local_msg(x)
        
        # Aggregate messages from neighbors
        messages = x_msg[src]  # [num_edges, hidden_channels]
        
        # Sum messages per destination node
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        
        # Normalize by degree (add small epsilon to avoid division by zero)
        degree = torch.bincount(dst, minlength=x.size(0)).float().unsqueeze(-1)
        out = out / (degree + 1e-8)
        
        return out
    
    def _global_attention(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Global attention via virtual super nodes.
        
        Steps:
        1. Pool nodes to create virtual super nodes (one per graph)
        2. Nodes attend to their graph's super node
        3. Broadcast updated super node info back to nodes
        """
        batch_size = batch.max().item() + 1
        
        # Create virtual nodes by pooling (one per graph)
        virtual_nodes = []
        for i in range(batch_size):
            graph_nodes = x[batch == i]
            virtual_node = graph_nodes.mean(dim=0)  # Mean pooling
            virtual_nodes.append(virtual_node)
        
        virtual_nodes = torch.stack(virtual_nodes)  # [batch_size, hidden_channels]
        virtual_nodes = self.virtual_proj(virtual_nodes)
        
        # Each node attends to its graph's virtual node
        # This is efficient: O(N) instead of O(N²)
        virtual_per_node = virtual_nodes[batch]  # [num_nodes, hidden_channels]
        
        # Attention: nodes (query) attend to virtual nodes (key, value)
        out = self.global_attention(
            query=x,
            key=virtual_per_node,
            value=virtual_per_node,
        )
        
        return out


class GOAT(BaseGraphTransformer):
    """
    GOAT: Global Transformer with O(N) complexity.
    
    Uses virtual nodes to approximate global attention efficiently.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of GOAT layers
        num_heads: Number of attention heads
        num_virtual_nodes: Number of virtual nodes per graph (default: 1)
        pe_dim: Positional encoding dimension
        dropout: Dropout rate
        task_type: 'graph_classification' or 'node_classification'
        pooling_type: Pooling for graph-level tasks ('mean', 'max', 'add')
    
    Example:
        >>> model = GOAT(
        ...     in_channels=32,
        ...     hidden_channels=256,
        ...     out_channels=2,
        ...     num_layers=4,
        ...     num_heads=8,
        ... )
        >>> data = Data(x=x, edge_index=edge_index, pe=pe, batch=batch)
        >>> out = model(data)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        num_virtual_nodes: int = 1,
        pe_dim: int = 8,
        dropout: float = 0.0,
        task_type: str = "graph_classification",
        pooling_type: str = "mean",
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            pe_dim=pe_dim,
            dropout=dropout,
            task_type=task_type,
        )
        
        self.num_heads = num_heads
        self.num_virtual_nodes = num_virtual_nodes
        
        # GOAT layers
        self.layers = nn.ModuleList([
            GOATLayer(
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                num_virtual_nodes=num_virtual_nodes,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.create_output_head(pooling_type)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data/Batch object with:
                - x: Node features [num_nodes, in_channels]
                - edge_index: Edge connectivity [2, num_edges]
                - pe: Positional encodings [num_nodes, pe_dim]
                - batch: Batch assignment [num_nodes] (for graph-level tasks)
                
        Returns:
            Output predictions [batch_size, out_channels] for graph-level
            or [num_nodes, out_channels] for node-level
        """
        # Encode input (features + positional encodings)
        x = self.encode_input(data)
        
        # Get batch assignment (for single graphs, create dummy batch)
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply GOAT layers
        for layer in self.layers:
            x = layer(x, data.edge_index, batch)
        
        # Output head
        if self.task_type == "graph_classification":
            # Pool to graph-level representation
            x = self.pool_graph(x, batch)
        
        # Final prediction
        out = self.output_head(x)
        
        return out
    
    def get_attention_weights(self, data: Data, layer_idx: int = 0) -> torch.Tensor:
        """
        Extract attention weights from a specific layer.
        
        Useful for visualization and analysis.
        
        Args:
            data: Input data
            layer_idx: Which layer to extract from
            
        Returns:
            Attention weights
        """
        # Run forward pass up to the specified layer
        x = self.encode_input(data)
        
        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, layer in enumerate(self.layers):
            if i == layer_idx:
                # Get attention weights (would need to modify layer to return them)
                # For now, just return a placeholder
                # TODO: Modify GOATLayer to optionally return attention weights
                pass
            x = layer(x, data.edge_index, batch)
        
        return None  # Placeholder

