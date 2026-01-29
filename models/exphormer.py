"""
Exphormer: Sparse Transformers for Graphs

Paper: Shirzad et al. "Exphormer: Sparse Transformers for Graphs" (ICML 2023)
Link: https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf

Key Ideas:
1. Use expander graphs to create sparse attention patterns
2. Expander graphs have good connectivity despite being sparse (d-regular)
3. Combine local (graph edges) + expander (virtual edges) attention
4. Complexity: O(N) for d-regular expanders vs O(N²) for full attention

Teacher's note: "poczytać o ekspanderach, dla macierzy laplaca są duże 
odległości między wartościami niezerowymi" - expanders are sparse but 
well-connected, perfect for sparse attention!

Expander Graphs:
- d-regular: every node has degree d
- Good expansion: information flows quickly
- Small diameter: log(N) despite sparsity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Tuple
import math

from .base import BaseGraphTransformer
from .layers import MultiHeadAttention, FeedForward


class ExpanderGraphBuilder:
    """
    Build expander graphs for sparse attention.
    
    We create a d-regular expander graph overlay on top of the input graph.
    This provides sparse but well-connected attention patterns.
    """
    
    @staticmethod
    def build_expander_edges(
        num_nodes: int,
        degree: int = 4,
        method: str = "random",
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Build expander graph edges.
        
        Args:
            num_nodes: Number of nodes
            degree: Degree of expander (d-regular)
            method: 'random', 'circulant', or 'ramanujan'
            device: Device to create tensor on
            
        Returns:
            edge_index: [2, num_edges] expander edges
        """
        if method == "random":
            return ExpanderGraphBuilder._random_expander(num_nodes, degree, device)
        elif method == "circulant":
            return ExpanderGraphBuilder._circulant_expander(num_nodes, degree, device)
        else:
            raise ValueError(f"Unknown expander method: {method}")
    
    @staticmethod
    def _random_expander(
        num_nodes: int,
        degree: int,
        device: str,
    ) -> torch.Tensor:
        """
        Random d-regular graph (simple construction).
        
        Not guaranteed to be an expander, but works well in practice.
        """
        edges = []
        
        for node in range(num_nodes):
            # Sample degree neighbors (without replacement)
            neighbors = torch.randperm(num_nodes)[:degree]
            
            for neighbor in neighbors:
                if neighbor != node:  # No self-loops
                    edges.append([node, neighbor.item()])
        
        edge_index = torch.tensor(edges, device=device).t()
        return edge_index
    
    @staticmethod
    def _circulant_expander(
        num_nodes: int,
        degree: int,
        device: str,
    ) -> torch.Tensor:
        """
        Circulant graph (guaranteed expander for certain parameters).
        
        Connect each node i to nodes (i + offset) % n for various offsets.
        """
        edges = []
        
        # Generate offsets (should be coprime to num_nodes for good expansion)
        offsets = []
        offset = 1
        while len(offsets) < degree:
            if math.gcd(offset, num_nodes) == 1:
                offsets.append(offset)
            offset += 1
        
        for node in range(num_nodes):
            for offset in offsets:
                neighbor = (node + offset) % num_nodes
                edges.append([node, neighbor])
        
        edge_index = torch.tensor(edges, device=device).t()
        return edge_index


class ExphormerLayer(nn.Module):
    """
    Single Exphormer layer with sparse attention.
    
    Combines:
    1. Local attention (on graph edges)
    2. Expander attention (on virtual expander edges)
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 8,
        expander_degree: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.expander_degree = expander_degree
        
        # Local attention (on graph edges)
        self.local_attention = MultiHeadAttention(
            hidden_channels,
            num_heads,
            dropout,
        )
        
        # Expander attention (on virtual expander edges)
        self.expander_attention = MultiHeadAttention(
            hidden_channels,
            num_heads,
            dropout,
        )
        
        # Feed-forward network
        self.ffn = FeedForward(hidden_channels, dropout=dropout)
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Learnable mixing parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        expander_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            edge_index: Local graph edges [2, num_edges]
            expander_edge_index: Expander graph edges [2, num_expander_edges]
            
        Returns:
            Updated node features [num_nodes, hidden_channels]
        """
        # 1. Local attention (on graph structure)
        x_norm = self.norm1(x)
        x_local = self.local_attention(
            x_norm, x_norm, x_norm,
            edge_index=edge_index,
        )
        
        # 2. Expander attention (on virtual expander edges)
        x_expander = self.expander_attention(
            x_norm, x_norm, x_norm,
            edge_index=expander_edge_index,
        )
        
        # Mix local and expander attention
        alpha = torch.sigmoid(self.alpha)  # Keep in [0, 1]
        x_combined = alpha * x_local + (1 - alpha) * x_expander
        
        x = x + self.dropout(x_combined)
        x = self.norm2(x)
        
        # 3. Feed-forward
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm3(x)
        
        return x


class Exphormer(BaseGraphTransformer):
    """
    Exphormer: Sparse transformer using expander graphs.
    
    Uses sparse attention on local graph + expander graph overlay.
    Achieves O(N) complexity with good global connectivity.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of Exphormer layers
        num_heads: Number of attention heads
        expander_degree: Degree of expander graph (d-regular)
        expander_method: Method to build expander ('random', 'circulant')
        pe_dim: Positional encoding dimension
        dropout: Dropout rate
        task_type: 'graph_classification' or 'node_classification'
        pooling_type: Pooling for graph-level tasks
    
    Example:
        >>> model = Exphormer(
        ...     in_channels=32,
        ...     hidden_channels=256,
        ...     out_channels=2,
        ...     num_layers=4,
        ...     expander_degree=4,
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
        expander_degree: int = 4,
        expander_method: str = "random",
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
        self.expander_degree = expander_degree
        self.expander_method = expander_method
        
        # Exphormer layers
        self.layers = nn.ModuleList([
            ExphormerLayer(
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                expander_degree=expander_degree,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.create_output_head(pooling_type)
        
        # Cache for expander edges (avoid recomputing)
        self._expander_cache = {}
    
    def _get_expander_edges(
        self,
        num_nodes: int,
        device: str,
    ) -> torch.Tensor:
        """
        Get or create expander edges for a graph of given size.
        
        Uses caching to avoid recomputing for same-sized graphs.
        """
        cache_key = (num_nodes, device)
        
        if cache_key not in self._expander_cache:
            expander_edges = ExpanderGraphBuilder.build_expander_edges(
                num_nodes=num_nodes,
                degree=self.expander_degree,
                method=self.expander_method,
                device=device,
            )
            self._expander_cache[cache_key] = expander_edges
        
        return self._expander_cache[cache_key]
    
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
            Output predictions
        """
        # Encode input (features + positional encodings)
        x = self.encode_input(data)
        
        # Get batch assignment
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Build expander edges for this batch
        expander_edges = self._build_batch_expander_edges(data, batch)
        
        # Apply Exphormer layers
        for layer in self.layers:
            x = layer(x, data.edge_index, expander_edges)
        
        # Output head
        if self.task_type == "graph_classification":
            # Pool to graph-level representation
            x = self.pool_graph(x, batch)
        
        # Final prediction
        out = self.output_head(x)
        
        return out
    
    def _build_batch_expander_edges(
        self,
        data: Data,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build expander edges for a batch of graphs.
        
        For each graph in the batch, create separate expander edges.
        """
        batch_size = batch.max().item() + 1
        device = data.x.device
        
        all_expander_edges = []
        
        for i in range(batch_size):
            # Get nodes for this graph
            mask = batch == i
            node_indices = torch.where(mask)[0]
            num_nodes = node_indices.size(0)
            
            # Get expander edges (indices within subgraph) [2, num_edges]
            expander_edges = self._get_expander_edges(num_nodes, device)
            expander_edges = expander_edges.long().to(device)
            
            # Remap to global indices: index each row so node_indices[expander_edges] stays (2, E)
            global_expander_edges = node_indices[expander_edges]
            
            all_expander_edges.append(global_expander_edges)
        
        # Concatenate all edges
        if len(all_expander_edges) > 0:
            expander_edges = torch.cat(all_expander_edges, dim=1)
        else:
            expander_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        return expander_edges
    
    def get_complexity_stats(self) -> dict:
        """
        Get theoretical complexity statistics.
        
        Returns:
            Dictionary with complexity info
        """
        return {
            "attention_complexity": "O(N * (d_graph + d_expander))",
            "expander_degree": self.expander_degree,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "hidden_channels": self.hidden_channels,
        }

