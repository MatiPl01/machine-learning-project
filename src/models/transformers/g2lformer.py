"""
G2LFormer: Global-to-Local Transformer for Graphs.

Compact implementation optimized for fast training.
Implements hybrid global-to-local attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, dim] or [num_nodes, dim]
            mask: Optional attention mask
        """
        # Handle both batched and unbatched inputs
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, num_nodes, dim]
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, dim = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)  # [batch, seq, dim]
        out = self.proj(out)

        if squeeze_output:
            out = out.squeeze(0)  # [num_nodes, dim]

        return out


class G2LFormerLayer(nn.Module):
    """Single G2LFormer layer with global and local attention."""

    def __init__(self, dim, num_heads=4, dropout=0.1, num_global_tokens=4):
        super().__init__()
        self.dim = dim
        self.num_global_tokens = num_global_tokens

        # Global attention (attends to all nodes)
        self.global_attn = MultiHeadAttention(dim, num_heads, dropout)

        # Local attention (GCN for neighbor aggregation)
        from torch_geometric.nn import GCNConv

        self.local_gcn = GCNConv(dim, dim)

        # Local attention (via graph structure)
        self.local_attn = nn.ModuleList(
            [
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
            ]
        )

        # Fusion
        self.fusion = nn.Linear(dim * 2, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, batch, global_tokens):
        """
        Args:
            x: Node features [num_nodes, dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch vector [num_nodes]
            global_tokens: Global tokens [batch_size, num_global_tokens, dim]
        """
        # Ensure edge_index is Long type (required by PyTorch Geometric)
        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        # Global attention: nodes attend to global tokens
        batch_size = global_tokens.size(0)
        num_nodes = x.size(0)

        # Convert to dense batch for attention
        x_dense, mask = to_dense_batch(x, batch)  # [batch, max_nodes, dim]
        global_tokens_expanded = global_tokens  # [batch, num_global, dim]

        # Concatenate global tokens with node features
        combined = torch.cat(
            [global_tokens_expanded, x_dense], dim=1
        )  # [batch, num_global+max_nodes, dim]
        combined_mask = torch.cat(
            [torch.ones(batch_size, global_tokens.size(1), device=x.device), mask],
            dim=1,
        )  # [batch, num_global+max_nodes]

        # Global attention
        combined_attn = self.global_attn(
            combined, mask=combined_mask.unsqueeze(1).unsqueeze(2)
        )

        # Split back
        global_tokens_new = combined_attn[:, : global_tokens.size(1), :]
        x_global = combined_attn[:, global_tokens.size(1) :, :]

        # Extract node features back (handle variable sizes)
        x_global_list = []
        for i in range(batch_size):
            node_mask = mask[i]
            num_valid_nodes = node_mask.sum().item()
            x_global_list.append(x_global[i, :num_valid_nodes, :])
        x_global = torch.cat(x_global_list, dim=0)  # [num_nodes, dim]

        # Local attention (simplified: aggregate from neighbors via edges)
        # Use GCN layer for local neighbor aggregation
        x_local = self.local_gcn(x, edge_index)

        # Fusion: combine global and local
        x_fused = torch.cat([x_global, x_local], dim=-1)
        x_fused = self.fusion(x_fused)
        x = self.norm1(x + x_fused)

        # Feed-forward
        x = self.norm2(x + self.ff(x))

        return x, global_tokens_new


class G2LFormer(nn.Module):
    """
    G2LFormer: Global-to-Local Transformer for Graphs.

    Compact version for fast training.

    Args:
        input_dim: Input node feature dimension
        hidden_dim: Hidden dimension (default: 64 for speed)
        output_dim: Output dimension
        num_layers: Number of transformer layers (default: 2 for speed)
        num_heads: Number of attention heads (default: 4)
        num_global_tokens: Number of global tokens (default: 4)
        dropout: Dropout rate (default: 0.1)
        task_type: 'classification' or 'regression'
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        output_dim=1,
        num_layers=2,
        num_heads=4,
        num_global_tokens=4,
        dropout=0.1,
        task_type="classification",
    ):
        super().__init__()
        self.task_type = task_type
        self.hidden_dim = hidden_dim
        self.num_global_tokens = num_global_tokens

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Global tokens (learnable)
        self.global_token_emb = nn.Parameter(
            torch.randn(1, num_global_tokens, hidden_dim)
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                G2LFormerLayer(hidden_dim, num_heads, dropout, num_global_tokens)
                for _ in range(num_layers)
            ]
        )

        # Graph-level pooling
        self.pool = lambda x, batch: torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )

        # Output head
        pool_dim = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.GELU(),
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

        batch_size = batch.max().item() + 1

        # Project input
        x = self.input_proj(x)  # [num_nodes, hidden_dim]

        # Initialize global tokens
        global_tokens = self.global_token_emb.expand(
            batch_size, -1, -1
        )  # [batch, num_global, hidden]

        # Apply transformer layers
        for layer in self.layers:
            x, global_tokens = layer(x, edge_index, batch, global_tokens)

        # Graph-level pooling
        x = self.pool(x, batch)  # [batch, pool_dim]

        # Output
        out = self.fc(x)

        if self.task_type == "classification" and out.size(1) == 1:
            # Binary classification: return logits (sigmoid applied in loss/metrics)
            return out
        elif self.task_type == "classification":
            return out
        else:
            return out
