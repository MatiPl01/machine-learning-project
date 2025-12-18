"""
Positional Encoding Utilities for Graph Transformers

Implements various positional encodings:
1. Laplacian Eigenvectors (spectral)
2. Random Walk (structural)
3. Degree Centrality (simple)

Based on:
- "Rethinking Graph Transformers with Spectral Attention" (NeurIPS 2021)
- "Graph Transformer for Graph-to-Sequence Learning" (AAAI 2020)
"""

import torch
import numpy as np
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import Data


def compute_laplacian_pe(data: Data, k: int = 8, normalization: str = "sym") -> torch.Tensor:
    """
    Compute Laplacian positional encodings using eigenvectors.
    
    The graph Laplacian encodes structural information. Its eigenvectors
    provide a spectral embedding of nodes. Teacher's note: "for Laplacian matrix 
    there are large distances between non-zero values" - this sparsity is key!
    
    Args:
        data: PyG Data object with edge_index
        k: Number of smallest non-trivial eigenvectors to use
        normalization: 'sym' for symmetric, 'rw' for random walk normalized
        
    Returns:
        Tensor of shape [num_nodes, k] with positional encodings
        
    Note:
        - Handles disconnected graphs gracefully
        - Uses smallest k eigenvectors (after constant eigenvector)
        - Sign ambiguity is handled by sorting
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # Compute Laplacian
    edge_index_lap, edge_weight = get_laplacian(
        edge_index, 
        normalization=normalization,
        num_nodes=num_nodes
    )
    
    # Convert to dense for eigendecomposition
    # Note: For large graphs, use sparse eigensolvers (scipy.sparse.linalg.eigsh)
    L = to_dense_adj(edge_index_lap, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
        # Sort by eigenvalues (ascending)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take k smallest non-trivial eigenvectors (skip the first constant one)
        # The first eigenvector is constant (all 1s) for connected graphs
        pe = eigenvectors[:, 1:k+1]
        
        # Handle case where graph has fewer than k+1 eigenvectors
        if pe.shape[1] < k:
            # Pad with zeros
            padding = torch.zeros(num_nodes, k - pe.shape[1], device=pe.device)
            pe = torch.cat([pe, padding], dim=1)
            
    except RuntimeError as e:
        print(f"Warning: Eigendecomposition failed, using zero PE. Error: {e}")
        pe = torch.zeros(num_nodes, k)
    
    return pe


def compute_random_walk_pe(data: Data, k: int = 8) -> torch.Tensor:
    """
    Compute Random Walk positional encodings.
    
    Uses powers of the transition matrix to capture multi-hop neighborhoods.
    Encodes structural similarity based on random walk probabilities.
    
    Args:
        data: PyG Data object with edge_index
        k: Number of random walk steps
        
    Returns:
        Tensor of shape [num_nodes, k] with RW landing probabilities
        
    Note:
        - Each column i contains the probability of landing at each node after i steps
        - Diagonal of A^k gives self-return probabilities
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # Get adjacency matrix
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    
    # Compute degree matrix for normalization
    deg = adj.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1  # Avoid division by zero for isolated nodes
    
    # Transition matrix: T = D^{-1} A
    T = adj / deg
    
    # Compute powers of transition matrix
    rw_pe = []
    T_power = torch.eye(num_nodes, device=T.device)
    
    for _ in range(k):
        T_power = T_power @ T
        # Extract diagonal (self-return probabilities)
        rw_pe.append(T_power.diagonal().unsqueeze(1))
    
    rw_pe = torch.cat(rw_pe, dim=1)
    
    return rw_pe


def compute_degree_centrality(data: Data, normalize: bool = True) -> torch.Tensor:
    """
    Compute degree centrality as a simple positional encoding.
    
    Args:
        data: PyG Data object with edge_index
        normalize: Whether to normalize by max degree
        
    Returns:
        Tensor of shape [num_nodes, 1] with degree centrality
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    
    # Compute node degrees
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    
    if normalize:
        max_deg = deg.max()
        if max_deg > 0:
            deg = deg / max_deg
    
    return deg.unsqueeze(1)


def add_positional_encodings(
    data: Data, 
    pe_type: str = "laplacian",
    pe_dim: int = 8,
    **kwargs
) -> Data:
    """
    Add positional encodings to a PyG Data object.
    
    Args:
        data: PyG Data object
        pe_type: Type of PE - 'laplacian', 'random_walk', 'degree', or 'none'
        pe_dim: Dimension of positional encoding
        **kwargs: Additional arguments for PE computation
        
    Returns:
        Data object with added 'pe' attribute
        
    Example:
        >>> data = Data(x=x, edge_index=edge_index)
        >>> data = add_positional_encodings(data, pe_type='laplacian', pe_dim=8)
        >>> print(data.pe.shape)  # [num_nodes, 8]
    """
    if pe_type == "laplacian":
        pe = compute_laplacian_pe(data, k=pe_dim, **kwargs)
    elif pe_type == "random_walk":
        pe = compute_random_walk_pe(data, k=pe_dim, **kwargs)
    elif pe_type == "degree":
        pe = compute_degree_centrality(data, **kwargs)
        # Replicate to match pe_dim
        if pe_dim > 1:
            pe = pe.repeat(1, pe_dim)
    elif pe_type == "none":
        pe = torch.zeros(data.num_nodes, pe_dim)
    else:
        raise ValueError(f"Unknown PE type: {pe_type}")
    
    data.pe = pe
    return data


def precompute_positional_encodings(dataset, pe_type: str = "laplacian", pe_dim: int = 8):
    """
    Precompute positional encodings for an entire dataset.
    
    Creates a wrapper that adds PE on-the-fly when accessing data.
    
    Args:
        dataset: PyG dataset
        pe_type: Type of PE to compute
        pe_dim: Dimension of PE
        
    Returns:
        Wrapped dataset with PE support
        
    Example:
        >>> dataset = ZINC(root='./data')
        >>> dataset = precompute_positional_encodings(dataset, pe_type='laplacian')
    """
    print(f"Setting up {pe_type} positional encodings (dim={pe_dim})...")
    
    # Create a wrapper class that adds PE on access
    class DatasetWithPE:
        def __init__(self, base_dataset, pe_type, pe_dim):
            self.base_dataset = base_dataset
            self.pe_type = pe_type
            self.pe_dim = pe_dim
            # Cache for computed PEs
            self._pe_cache = {}
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            # Handle tensor indices (from split_idx)
            if isinstance(idx, torch.Tensor):
                return [self[i.item()] for i in idx]
            
            # Handle list/array indices
            if isinstance(idx, (list, tuple)):
                return [self[i] for i in idx]
            
            # Single integer index
            if idx not in self._pe_cache:
                data = self.base_dataset[idx]
                self._pe_cache[idx] = add_positional_encodings(
                    data, pe_type=self.pe_type, pe_dim=self.pe_dim
                )
            
            return self._pe_cache[idx]
        
        def __getattr__(self, name):
            # Delegate other attributes to base dataset
            return getattr(self.base_dataset, name)
    
    wrapped_dataset = DatasetWithPE(dataset, pe_type, pe_dim)
    
    # Precompute PEs by accessing all data
    print("Precomputing encodings...")
    for i in range(len(wrapped_dataset)):
        _ = wrapped_dataset[i]
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} graphs")
    
    print("Done!")
    return wrapped_dataset

