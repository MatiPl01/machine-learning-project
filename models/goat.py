import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional

from .base import BaseGraphTransformer, global_mean_pool
from .layers import MultiHeadAttention, FeedForward


class GOATLayer(nn.Module):
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

        self.local_msg = nn.Linear(hidden_channels, hidden_channels)
        self.global_attention = MultiHeadAttention(hidden_channels, num_heads, dropout)
        self.virtual_proj = nn.Linear(hidden_channels, hidden_channels)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        self.ffn = FeedForward(hidden_channels, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x_local = self._local_message_passing(x, edge_index)
        x = x + self.dropout(x_local)
        x = self.norm1(x)
        x_global = self._global_attention(x, batch)
        x = x + self.dropout(x_global)
        x = self.norm2(x)
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm3(x)
        return x

    def _local_message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        x_msg = self.local_msg(x)
        messages = x_msg[src]
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        degree = torch.bincount(dst, minlength=x.size(0)).float().unsqueeze(-1)
        out = out / (degree + 1e-8)
        return out

    def _global_attention(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        virtual_nodes = []
        for i in range(batch_size):
            graph_nodes = x[batch == i]
            virtual_node = graph_nodes.mean(dim=0)
            virtual_nodes.append(virtual_node)
        virtual_nodes = torch.stack(virtual_nodes)
        virtual_nodes = self.virtual_proj(virtual_nodes)
        virtual_per_node = virtual_nodes[batch]
        out = self.global_attention(
            query=x,
            key=virtual_per_node,
            value=virtual_per_node,
        )
        return out


class GOAT(BaseGraphTransformer):
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

        self.layers = nn.ModuleList([
            GOATLayer(
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                num_virtual_nodes=num_virtual_nodes,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.create_output_head(pooling_type)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.encode_input(data)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for layer in self.layers:
            x = layer(x, data.edge_index, batch)

        if self.task_type == "graph_classification":
            x = self.pool_graph(x, batch)
        return self.output_head(x)

    def get_attention_weights(self, data: Data, layer_idx: int = 0) -> torch.Tensor:
        x = self.encode_input(data)
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for i, layer in enumerate(self.layers):
            if i == layer_idx:
                pass
            x = layer(x, data.edge_index, batch)
        return None
