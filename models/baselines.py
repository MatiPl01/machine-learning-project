import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from typing import Optional


class GCN(nn.Module):
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

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

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
        x, edge_index = data.x, data.edge_index
        if x.dtype != torch.float32:
            x = x.float()

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task_type == "graph_classification":
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self._pool_graph(x, batch)

        return self.output_head(x)

    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class GAT(nn.Module):
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

        assert hidden_channels % num_heads == 0
        head_dim = hidden_channels // num_heads

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.convs.append(GATConv(in_channels, head_dim, heads=num_heads, dropout=dropout, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, head_dim, heads=num_heads, dropout=dropout, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels, head_dim, heads=num_heads, dropout=dropout, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

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
        x, edge_index = data.x, data.edge_index
        if x.dtype != torch.float32:
            x = x.float()

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task_type == "graph_classification":
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self._pool_graph(x, batch)

        return self.output_head(x)

    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class GraphMLP(nn.Module):
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

        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

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
        x = data.x
        if x.dtype != torch.float32:
            x = x.float()
        x = self.mlp(x)
        if self.task_type == "graph_classification":
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self._pool_graph(x, batch)
        return self.output_head(x)

    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class GIN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
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

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        mlp_input = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.convs.append(GINConv(mlp_input, train_eps=train_eps))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

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
        x, edge_index = data.x, data.edge_index
        if x.dtype != torch.float32:
            x = x.float()

        layer_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        if self.task_type == "graph_classification":
            batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                    torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pooled_outputs = [self._pool_graph(layer_out, batch) for layer_out in layer_outputs]
            x = torch.cat(pooled_outputs, dim=-1)

        return self.output_head(x)

    def _pool_graph(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        elif self.pooling_type == "add":
            return global_add_pool(x, batch)
        raise ValueError(f"Unknown pooling type: {self.pooling_type}")
