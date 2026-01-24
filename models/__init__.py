"""
Graph Transformer Models

This package contains implementations of:

Graph Transformers:
- GOAT: Global attention transformer with virtual nodes
- Exphormer: Sparse transformer with expander graphs

Baseline GNN Models:
- GCN: Graph Convolutional Network (Kipf & Welling, 2017)
- GAT: Graph Attention Network (Veličković et al., 2018)
- GIN: Graph Isomorphism Network (Xu et al., 2019)
- GraphMLP: Simple MLP baseline (no graph structure)

Hybrid Models:
- GCNVirtualNode: GCN with Virtual Node for global context
- GINVirtualNode: GIN with Virtual Node (strongest hybrid)
"""

from .goat import GOAT
from .exphormer import Exphormer
from .baselines import GCN, GAT, GIN, GraphMLP
from .hybrid import GCNVirtualNode, GINVirtualNode

__all__ = [
    # Graph Transformers
    "GOAT",
    "Exphormer",
    # Baseline GNNs
    "GCN",
    "GAT",
    "GIN",
    "GraphMLP",
    # Hybrid Models
    "GCNVirtualNode",
    "GINVirtualNode",
]


