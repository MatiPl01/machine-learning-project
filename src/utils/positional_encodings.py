import torch
import numpy as np
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.data import Data


def compute_laplacian_pe(data: Data, k: int = 8, normalization: str = "sym") -> torch.Tensor:
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_index_lap, edge_weight = get_laplacian(
        edge_index,
        normalization=normalization,
        num_nodes=num_nodes
    )
    L = to_dense_adj(edge_index_lap, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(L)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        pe = eigenvectors[:, 1:k+1]
        if pe.shape[1] < k:
            padding = torch.zeros(num_nodes, k - pe.shape[1], device=pe.device)
            pe = torch.cat([pe, padding], dim=1)
    except RuntimeError:
        pe = torch.zeros(num_nodes, k)
    return pe


def compute_random_walk_pe(data: Data, k: int = 8) -> torch.Tensor:
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    deg = adj.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1
    T = adj / deg
    rw_pe = []
    T_power = torch.eye(num_nodes, device=T.device)
    for _ in range(k):
        T_power = T_power @ T
        rw_pe.append(T_power.diagonal().unsqueeze(1))
    return torch.cat(rw_pe, dim=1)


def compute_degree_centrality(data: Data, normalize: bool = True) -> torch.Tensor:
    num_nodes = data.num_nodes
    edge_index = data.edge_index
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
    if pe_type == "laplacian":
        pe = compute_laplacian_pe(data, k=pe_dim, **kwargs)
    elif pe_type == "random_walk":
        pe = compute_random_walk_pe(data, k=pe_dim, **kwargs)
    elif pe_type == "degree":
        pe = compute_degree_centrality(data, **kwargs)
        if pe_dim > 1:
            pe = pe.repeat(1, pe_dim)
    elif pe_type == "none":
        pe = torch.zeros(data.num_nodes, pe_dim)
    else:
        raise ValueError(f"Unknown PE type: {pe_type}")
    data.pe = pe
    return data


def precompute_positional_encodings(dataset, pe_type: str = "laplacian", pe_dim: int = 8):
    print(f"Setting up {pe_type} positional encodings (dim={pe_dim})...")

    class DatasetWithPE:
        def __init__(self, base_dataset, pe_type, pe_dim):
            self.base_dataset = base_dataset
            self.pe_type = pe_type
            self.pe_dim = pe_dim
            self._pe_cache = {}

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            if isinstance(idx, torch.Tensor):
                return [self[i.item()] for i in idx]
            if isinstance(idx, (list, tuple)):
                return [self[i] for i in idx]
            if idx not in self._pe_cache:
                data = self.base_dataset[idx]
                self._pe_cache[idx] = add_positional_encodings(
                    data, pe_type=self.pe_type, pe_dim=self.pe_dim
                )
            return self._pe_cache[idx]

        def __getattr__(self, name):
            return getattr(self.base_dataset, name)

    wrapped_dataset = DatasetWithPE(dataset, pe_type, pe_dim)
    print("Precomputing encodings...")
    for i in range(len(wrapped_dataset)):
        _ = wrapped_dataset[i]
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} graphs")
    print("Done!")
    return wrapped_dataset
