import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from typing import Optional


class VirtualNodeLayer(nn.Module):
    def __init__(self, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
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
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        virtual_node: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple:
        virtual_to_nodes = virtual_node[batch]
        x = x + virtual_to_nodes
        node_to_virtual = global_mean_pool(x, batch)
        virtual_node_updated = self.virtual_mlp(node_to_virtual)
        alpha = torch.sigmoid(self.residual_weight)
        virtual_node = alpha * virtual_node + (1 - alpha) * virtual_node_updated
        return x, virtual_node


class GCNVirtualNode(nn.Module):
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

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.virtual_node_embedding = nn.Embedding(1, hidden_channels)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.virtual_node_layers = nn.ModuleList([
            VirtualNodeLayer(hidden_channels, dropout)
            for _ in range(num_layers - 1)
        ])

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

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        batch_size = batch.max().item() + 1

        x = self.input_proj(x)
        virtual_node = self.virtual_node_embedding.weight.expand(batch_size, -1)

        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_conv = conv(x, edge_index)
            x_conv = bn(x_conv)
            x_conv = F.relu(x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            x = x + x_conv
            if i < len(self.virtual_node_layers):
                x, virtual_node = self.virtual_node_layers[i](x, virtual_node, batch)

        if self.task_type == "graph_classification":
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


class GINVirtualNode(nn.Module):
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

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.virtual_node_embedding = nn.Embedding(1, hidden_channels)

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

        self.virtual_node_layers = nn.ModuleList([
            VirtualNodeLayer(hidden_channels, dropout)
            for _ in range(num_layers - 1)
        ])

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

        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        batch_size = batch.max().item() + 1

        x = self.input_proj(x)
        virtual_node = self.virtual_node_embedding.weight.expand(batch_size, -1)
        layer_outputs = []

        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_conv = conv(x, edge_index)
            x_conv = bn(x_conv)
            x_conv = F.relu(x_conv)
            x_conv = F.dropout(x_conv, p=self.dropout, training=self.training)
            x = x + x_conv
            if i < len(self.virtual_node_layers):
                x, virtual_node = self.virtual_node_layers[i](x, virtual_node, batch)
            layer_outputs.append(x)

        if self.task_type == "graph_classification":
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
