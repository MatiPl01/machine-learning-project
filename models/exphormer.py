import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional, Tuple
import math

from .base import BaseGraphTransformer
from .layers import MultiHeadAttention, FeedForward


class ExpanderGraphBuilder:
    @staticmethod
    def build_expander_edges(
        num_nodes: int,
        degree: int = 4,
        method: str = "random",
        device: str = "cpu",
    ) -> torch.Tensor:
        if method == "random":
            return ExpanderGraphBuilder._random_expander(num_nodes, degree, device)
        elif method == "circulant":
            return ExpanderGraphBuilder._circulant_expander(num_nodes, degree, device)
        raise ValueError(f"Unknown expander method: {method}")

    @staticmethod
    def _random_expander(num_nodes: int, degree: int, device: str) -> torch.Tensor:
        edges = []
        for node in range(num_nodes):
            neighbors = torch.randperm(num_nodes)[:degree]
            for neighbor in neighbors:
                if neighbor != node:
                    edges.append([node, neighbor.item()])
        return torch.tensor(edges, dtype=torch.long, device=device).t()

    @staticmethod
    def _circulant_expander(num_nodes: int, degree: int, device: str) -> torch.Tensor:
        edges = []
        offsets = []
        offset = 1
        while len(offsets) < degree:
            if math.gcd(offset, num_nodes) == 1:
                offsets.append(offset)
            offset += 1
        for node in range(num_nodes):
            for off in offsets:
                neighbor = (node + off) % num_nodes
                edges.append([node, neighbor])
        return torch.tensor(edges, dtype=torch.long, device=device).t()


class ExphormerLayer(nn.Module):
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

        self.local_attention = MultiHeadAttention(hidden_channels, num_heads, dropout)
        self.expander_attention = MultiHeadAttention(hidden_channels, num_heads, dropout)
        self.ffn = FeedForward(hidden_channels, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        expander_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        x_local = self.local_attention(x_norm, x_norm, x_norm, edge_index=edge_index)
        x_expander = self.expander_attention(x_norm, x_norm, x_norm, edge_index=expander_edge_index)
        alpha = torch.sigmoid(self.alpha)
        x_combined = alpha * x_local + (1 - alpha) * x_expander
        x = x + self.dropout(x_combined)
        x = self.norm2(x)
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm3(x)
        return x


class Exphormer(BaseGraphTransformer):
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

        self.layers = nn.ModuleList([
            ExphormerLayer(
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                expander_degree=expander_degree,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.create_output_head(pooling_type)
        self._expander_cache = {}

    def _get_expander_edges(self, num_nodes: int, device: str) -> torch.Tensor:
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
        x = self.encode_input(data)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        expander_edges = self._build_batch_expander_edges(data, batch)

        for layer in self.layers:
            x = layer(x, data.edge_index, expander_edges)

        if self.task_type == "graph_classification":
            x = self.pool_graph(x, batch)
        return self.output_head(x)

    def _build_batch_expander_edges(self, data: Data, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        device = data.x.device
        all_expander_edges = []
        for i in range(batch_size):
            mask = (batch == i)
            node_indices = torch.where(mask)[0].long().to(device)
            num_nodes = node_indices.size(0)
            expander_edges = self._get_expander_edges(num_nodes, device)
            expander_edges = expander_edges.long().to(device)
            global_expander_edges = node_indices[expander_edges]
            all_expander_edges.append(global_expander_edges)
        if all_expander_edges:
            return torch.cat(all_expander_edges, dim=1)
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    def get_complexity_stats(self) -> dict:
        return {
            "attention_complexity": "O(N * (d_graph + d_expander))",
            "expander_degree": self.expander_degree,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "hidden_channels": self.hidden_channels,
        }
