import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Optional


class BaseGraphTransformer(nn.Module):
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

        self.node_encoder = nn.Linear(in_channels + pe_dim, hidden_channels)
        self.dropout_layer = nn.Dropout(dropout)
        self.output_head = None

    def create_output_head(self, pooling_type: str = "mean"):
        if self.task_type == "graph_classification":
            self.pooling_type = pooling_type
            self.output_head = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.out_channels)
            )
        elif self.task_type == "node_classification":
            self.output_head = nn.Linear(self.hidden_channels, self.out_channels)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    def encode_input(self, data: Data) -> torch.Tensor:
        if hasattr(data, 'pe') and data.pe is not None:
            x = torch.cat([data.x, data.pe], dim=-1)
        else:
            pe = torch.zeros(data.x.size(0), self.pe_dim, device=data.x.device)
            x = torch.cat([data.x, pe], dim=-1)
        x = self.node_encoder(x)
        x = self.dropout_layer(x)
        return x

    def forward(self, data: Data) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    batch_size = batch.max().item() + 1
    return torch.stack([x[batch == i].mean(0) for i in range(batch_size)])


def global_max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    batch_size = batch.max().item() + 1
    return torch.stack([x[batch == i].max(0)[0] for i in range(batch_size)])


def global_add_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    batch_size = batch.max().item() + 1
    return torch.stack([x[batch == i].sum(0) for i in range(batch_size)])
