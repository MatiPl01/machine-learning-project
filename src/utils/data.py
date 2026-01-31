import torch
import numpy as np
import torch_geometric

torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.data.Data,
        torch_geometric.data.storage.GlobalStorage,
        torch_geometric.data.storage.NodeStorage,
        torch_geometric.data.storage.EdgeStorage,
    ]
)


def create_train_val_test_split(dataset, train_ratio=0.8, val_ratio=0.1):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    indices = torch.randperm(total_size)
    split_idx = {
        "train": indices[:train_size],
        "valid": indices[train_size : train_size + val_size],
        "test": indices[train_size + val_size :],
    }
    return split_idx


def load_molhiv_dataset(root="./data"):
    from ogb.graphproppred import PygGraphPropPredDataset
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    split_idx = dataset.get_idx_split()
    return dataset, split_idx


def load_molpcba_dataset(root="./data"):
    from ogb.graphproppred import PygGraphPropPredDataset
    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=root)
    split_idx = dataset.get_idx_split()
    return dataset, split_idx


def load_ppa_dataset(root="./data"):
    from ogb.graphproppred import PygGraphPropPredDataset
    dataset = PygGraphPropPredDataset(name="ogbg-ppa", root=root)
    split_idx = dataset.get_idx_split()
    return dataset, split_idx


def load_peptides_func_dataset(root="./data"):
    from torch_geometric.datasets import LRGBDataset
    dataset = LRGBDataset(name="peptides-func", root=root)
    split_idx = create_train_val_test_split(dataset)
    return dataset, split_idx


def load_zinc_dataset(root="./data"):
    from torch_geometric.datasets import ZINC
    dataset = ZINC(root=root, subset=True, split="train")
    split_idx = create_train_val_test_split(dataset)
    return dataset, split_idx


def analyze_graph_properties(graphs):
    num_nodes = []
    num_edges = []
    node_degrees = []
    for graph in graphs:
        num_nodes.append(graph.num_nodes)
        num_edges.append(graph.num_edges)
        if hasattr(graph, "edge_index") and graph.edge_index.size(1) > 0:
            degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
            degrees.scatter_add_(
                0,
                graph.edge_index[0],
                torch.ones(graph.edge_index.size(1), dtype=torch.long),
            )
            node_degrees.extend(degrees.tolist())
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "node_degrees": node_degrees,
    }


def print_dataset_summary(dataset, sample_size=50):
    print("=" * 60)
    dataset_name = getattr(dataset, "name", dataset.__class__.__name__)
    print(f"DATASET SUMMARY: {dataset_name.upper()}")
    print("=" * 60)
    sample_graphs = [dataset[i] for i in range(min(sample_size, len(dataset)))]
    props = analyze_graph_properties(sample_graphs)
    print(f"Total graphs: {len(dataset):,}")
    print(f"Avg nodes: {np.mean(props['num_nodes']):.1f} ± {np.std(props['num_nodes']):.1f}")
    print(f"Avg edges: {np.mean(props['num_edges']):.1f} ± {np.std(props['num_edges']):.1f}")
    print(f"Avg degree: {np.mean(props['node_degrees']):.1f} ± {np.std(props['node_degrees']):.1f}")
    print(f"Node range: {min(props['num_nodes'])} - {max(props['num_nodes'])}")
    print(f"Edge range: {min(props['num_edges'])} - {max(props['num_edges'])}")
    print("=" * 60)
