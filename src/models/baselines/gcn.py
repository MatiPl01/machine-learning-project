"""
Graph Convolutional Network (GCN) baseline model.

Compact implementation for fast training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCN(nn.Module):
    """
    Simple GCN baseline with 2-3 layers.

    Args:
        input_dim: Input node feature dimension
        hidden_dim: Hidden layer dimension (default: 64 for speed)
        output_dim: Output dimension (task-specific)
        num_layers: Number of GCN layers (default: 2 for shallow baseline)
        dropout: Dropout rate (default: 0.5)
        task_type: 'classification' or 'regression'
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
        dropout=0.5,
        task_type="classification",
    ):
        super().__init__()
        self.task_type = task_type
        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers (disable caching to avoid dtype issues)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=False))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=False))

        # Graph-level pooling: mean + max
        self.pool = lambda x, batch: torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )

        # Output head
        pool_dim = hidden_dim * 2  # mean + max pooling
        self.fc = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes]
        """
        # Ensure edge_index is Long type and contiguous (required by PyTorch Geometric)
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()
        edge_index = edge_index.contiguous()

        # Ensure x is float (should be, but double-check to avoid dtype issues)
        if x.dtype != torch.float32 and x.dtype != torch.float64:
            x = x.float()

        # GCN layers
        for i, conv in enumerate(self.convs):
            # Clear any cached normalization to force fresh computation
            if hasattr(conv, "_cached_edge_index"):
                conv._cached_edge_index = None
            x = conv(x, edge_index)
            # Ensure output is float after each layer (safety check)
            if x.dtype != torch.float32 and x.dtype != torch.float64:
                x = x.float()
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        x = self.pool(x, batch)

        # Output
        out = self.fc(x)

        if self.task_type == "classification" and out.size(1) == 1:
            # Binary classification: return logits (sigmoid applied in loss/metrics)
            return out
        elif self.task_type == "classification":
            # Multi-class: logits (use with CrossEntropyLoss)
            return out
        else:
            # Regression: direct output
            return out
