"""
Model implementations for Graph Transformers project.
Includes G2LFormer, GCN, and GAT baseline models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


class PositionalEncoding(nn.Module):
    """Positional encoding using Laplacian eigenvectors."""

    def __init__(self, dim, max_nodes=512):
        super().__init__()
        self.dim = dim

    def forward(self, x, edge_index):
        """Compute positional encoding from graph structure."""
        # Simple learnable positional encoding
        # In practice, you'd compute Laplacian eigenvectors here
        num_nodes = x.size(0)
        pos_enc = torch.randn(num_nodes, self.dim, device=x.device)
        return pos_enc


class GCNBaseline(nn.Module):
    """Graph Convolutional Network baseline."""

    def __init__(
        self, input_dim, hidden_dim=64, num_layers=3, output_dim=1, dropout=0.5
    ):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Convert input to float if needed
        if x.dtype != torch.float32:
            x = x.float()

        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Final prediction
        x = self.lin(x)
        return x


class GATBaseline(nn.Module):
    """Graph Attention Network baseline."""

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        output_dim=1,
        dropout=0.5,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(
                input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=True
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )

        # Output layer
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    heads=1,
                    dropout=dropout,
                    concat=False,
                )
            )

        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Convert input to float if needed
        if x.dtype != torch.float32:
            x = x.float()

        # Graph attention layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = global_mean_pool(x, batch)

        # Final prediction
        x = self.lin(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape

        # Linear projections
        Q = (
            self.q_lin(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_lin(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_lin(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.out_lin(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward."""

    def __init__(self, dim, num_heads=8, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4

        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Self-attention with residual
        x = self.norm1(x + self.attn(x, mask))
        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))
        return x


class G2LFormer(nn.Module):
    """
    G2LFormer: Global-to-Local Attention Scheme in Graph Transformers.
    Implements hybrid attention combining global and local patterns.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        output_dim=1,
        dropout=0.1,
        use_pos_encoding=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_pos_encoding = use_pos_encoding

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim)

        # Global token (learnable)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # Local attention (GAT-like) for local neighborhoods
        self.local_attn = GATConv(
            hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index, batch):
        # Convert input to float if needed
        if x.dtype != torch.float32:
            x = x.float()

        # Project input
        x = self.input_proj(x)

        # Add positional encoding if enabled
        if self.use_pos_encoding:
            pos_enc = self.pos_encoding(x, edge_index)
            x = x + pos_enc

        # Convert to dense batch format for transformer
        x_dense, mask = to_dense_batch(x, batch)
        batch_size, max_nodes, hidden_dim = x_dense.shape

        # Add global token
        global_tokens = self.global_token.expand(batch_size, 1, hidden_dim)
        x_with_global = torch.cat([global_tokens, x_dense], dim=1)

        # Create mask for global token (always visible)
        if mask is not None:
            global_mask = torch.ones(
                batch_size, 1, device=mask.device, dtype=mask.dtype
            )
            mask_with_global = torch.cat([global_mask, mask], dim=1)
            mask_with_global = mask_with_global.unsqueeze(1).unsqueeze(2)
        else:
            mask_with_global = None

        # Apply transformer blocks (global attention)
        for transformer in self.transformer_blocks:
            x_with_global = transformer(x_with_global, mask_with_global)

        # Extract global token and node features
        global_features = x_with_global[:, 0, :]  # [batch_size, hidden_dim]
        node_features = x_with_global[:, 1:, :]  # [batch_size, max_nodes, hidden_dim]

        # Convert back to sparse format for local attention
        # Use the global token features to enhance node features
        node_features_sparse = node_features[mask.bool()]  # [total_nodes, hidden_dim]

        # Apply local attention (GAT) to capture neighborhood patterns
        node_features_local = self.local_attn(node_features_sparse, edge_index)

        # Combine global and local information
        # For each graph, pool node features
        x_pooled = global_mean_pool(node_features_local, batch)

        # Combine with global token
        combined = x_pooled + global_features

        # Final prediction
        out = self.output_proj(combined)
        return out


def get_model(model_name, input_dim, output_dim=1, **kwargs):
    """Factory function to get a model by name."""
    models = {
        "gcn": GCNBaseline,
        "gat": GATBaseline,
        "g2lformer": G2LFormer,
    }

    if model_name.lower() not in models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(models.keys())}"
        )

    ModelClass = models[model_name.lower()]
    return ModelClass(input_dim=input_dim, output_dim=output_dim, **kwargs)
