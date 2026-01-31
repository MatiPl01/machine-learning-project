import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert hidden_channels % num_heads == 0
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.out_proj = nn.Linear(hidden_channels, hidden_channels, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)

        if edge_index is not None:
            out = self._sparse_attention(Q, K, V, edge_index)
        else:
            out = self._full_attention(Q, K, V, attention_mask)

        out = out.view(batch_size, self.hidden_channels)
        out = self.out_proj(out)
        out = self.dropout_layer(out)
        return out

    def _full_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_scores = torch.einsum('bhd,bHd->bhH', Q, K) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        out = torch.einsum('bhH,bHd->bhd', attn_weights, V)
        return out

    def _sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        attn_scores = (Q[dst] * K[src]).sum(dim=-1) * self.scale
        attn_weights = self._softmax_per_node(attn_scores, dst, Q.size(0))
        attn_weights = self.dropout_layer(attn_weights)
        weighted_values = attn_weights.unsqueeze(-1) * V[src]
        out = torch.zeros_like(V)
        out.index_add_(0, dst, weighted_values)
        return out

    def _softmax_per_node(
        self,
        scores: torch.Tensor,
        index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        max_scores = torch.full((num_nodes, self.num_heads), float('-inf'), device=scores.device)
        max_scores.scatter_reduce_(0, index.unsqueeze(-1).expand_as(scores), scores, reduce='amax')
        max_scores = max_scores[index]
        exp_scores = (scores - max_scores).exp()
        sum_exp = torch.zeros(num_nodes, self.num_heads, device=scores.device)
        sum_exp.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)
        sum_exp = sum_exp[index]
        return exp_scores / (sum_exp + 1e-8)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        ffn_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if ffn_channels is None:
            ffn_channels = hidden_channels * 4
        self.fc1 = nn.Linear(hidden_channels, ffn_channels)
        self.fc2 = nn.Linear(ffn_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 8,
        ffn_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_channels, num_heads, dropout)
        self.ffn = FeedForward(hidden_channels, ffn_channels, dropout)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out = self.attention(x_norm, x_norm, x_norm, attention_mask, edge_index)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x
