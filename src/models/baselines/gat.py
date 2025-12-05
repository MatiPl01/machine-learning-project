"""
Graph Attention Network (GAT) baseline model.

Compact implementation for fast training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GAT(nn.Module):
    """
    Simple GAT baseline with attention mechanism.

    Args:
        input_dim: Input node feature dimension
        hidden_dim: Hidden layer dimension (default: 64 for speed)
        output_dim: Output dimension (task-specific)
        num_layers: Number of GAT layers (default: 2 for shallow baseline)
        num_heads: Number of attention heads (default: 2)
        dropout: Dropout rate (default: 0.5)
        task_type: 'classification' or 'regression'
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
        num_heads=2,
        dropout=0.5,
        task_type="classification",
    ):
        super().__init__()
        self.task_type = task_type
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout
                )
            )
        if num_layers > 1:
            # Last layer: single head for output dimension consistency
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
            )

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
        # Ensure edge_index is Long type (required by PyTorch Geometric)
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        # GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
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
