import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import warnings
import numpy as np
from .constants import DATASET_COLORS, DISTINCT_COLORS

warnings.filterwarnings("ignore")


def get_dataset_color(dataset_name, index=None):
    if dataset_name in DATASET_COLORS:
        return DATASET_COLORS[dataset_name]
    if index is not None:
        return DISTINCT_COLORS[index % len(DISTINCT_COLORS)]
    return DISTINCT_COLORS[0]


def get_dataset_name(dataset):
    return (
        dataset.name
        if hasattr(dataset, "name") and dataset.name
        else dataset.__class__.__name__
    )


def visualize_sample_graphs(datasets, num_samples=4, figsize=(20, 15)):
    _, axes = plt.subplots(len(datasets), num_samples, figsize=figsize)
    if len(datasets) == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, dataset in enumerate(datasets):
        name = get_dataset_name(dataset)
        graph_sizes = [(j, dataset[j].num_nodes) for j in range(len(dataset))]
        graph_sizes.sort(key=lambda x: x[1])
        sample_indices = [graph_sizes[0][0], graph_sizes[-1][0]]
        sample_labels = ["Smallest", "Largest"]
        import random
        random.seed(42)
        for j in range(num_samples - 2):
            random_idx = random.randint(0, len(dataset) - 1)
            sample_indices.append(random_idx)
            sample_labels.append(f"Random {j+1}")

        for j, (graph_idx, label) in enumerate(zip(sample_indices, sample_labels)):
            graph = dataset[graph_idx]
            try:
                G = to_networkx(graph, to_undirected=True)
                pos = nx.spring_layout(G, k=1, iterations=50)
                nx.draw(
                    G,
                    pos,
                    ax=axes[i, j],
                    node_size=50,
                    node_color="lightblue",
                    edge_color="gray",
                    with_labels=False,
                    alpha=0.8,
                )
                axes[i, j].set_title(
                    f"{name}\n{label}\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}"
                )
            except (ValueError, RuntimeError):
                axes[i, j].text(
                    0.5, 0.5, "Visualization Error",
                    ha="center", va="center", transform=axes[i, j].transAxes,
                )
                axes[i, j].set_title(f"{name}\n{label}")
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()


def plot_graph_size_distributions(datasets, figsize=(15, 5)):
    _, axes = plt.subplots(1, 2, figsize=figsize)
    all_data = {}
    for dataset in datasets:
        name = get_dataset_name(dataset)
        sample_graphs = [dataset[i] for i in range(min(500, len(dataset)))]
        num_nodes = [g.num_nodes for g in sample_graphs]
        num_edges = [g.num_edges for g in sample_graphs]
        all_data[name] = {"num_nodes": num_nodes, "num_edges": num_edges}

    ax = axes[0]
    for i, (name, data) in enumerate(all_data.items()):
        color = get_dataset_color(name, i)
        ax.hist(data["num_nodes"], alpha=0.7, label=name, bins=30, color=color)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Frequency")
    ax.set_title("Node Count Distribution")
    ax.legend()
    ax.set_yscale("linear")

    ax = axes[1]
    for i, (name, data) in enumerate(all_data.items()):
        color = get_dataset_color(name, i)
        ax.hist(data["num_edges"], alpha=0.7, label=name, bins=30, color=color)
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Frequency")
    ax.set_title("Edge Count Distribution")
    ax.legend()
    ax.set_yscale("linear")
    plt.tight_layout()
    plt.show()


def analyze_graph_properties(datasets, sample_size=100):
    results = {}
    for dataset in datasets:
        name = get_dataset_name(dataset)
        print(f"\nAnalyzing {name}...")
        sample_indices = np.random.choice(
            len(dataset), min(sample_size, len(dataset)), replace=False
        )
        sample_graphs = [dataset[i] for i in sample_indices]
        num_nodes = [g.num_nodes for g in sample_graphs]
        num_edges = [g.num_edges for g in sample_graphs]
        diameters = []
        connectivity_ratios = []
        shortest_paths = []

        for graph in sample_graphs:
            try:
                G = to_networkx(graph, to_undirected=True)
                if nx.is_connected(G):
                    diameter = nx.diameter(G)
                    diameters.append(diameter)
                else:
                    largest_cc = max(nx.connected_components(G), key=len)
                    G_largest = G.subgraph(largest_cc)
                    diameter = nx.diameter(G_largest)
                    diameters.append(diameter)
                n = G.number_of_nodes()
                max_edges = n * (n - 1) / 2 if n > 1 else 0
                connectivity_ratio = G.number_of_edges() / max_edges if max_edges > 0 else 0
                connectivity_ratios.append(connectivity_ratio)
                if G.number_of_nodes() <= 100:
                    path_lengths = []
                    for u in G.nodes():
                        for v in G.nodes():
                            if u != v and nx.has_path(G, u, v):
                                path_lengths.append(nx.shortest_path_length(G, u, v))
                    shortest_paths.extend(path_lengths)
                else:
                    nodes = list(G.nodes())
                    for _ in range(min(1000, len(nodes) ** 2)):
                        u, v = np.random.choice(nodes, 2, replace=False)
                        if nx.has_path(G, u, v):
                            shortest_paths.append(nx.shortest_path_length(G, u, v))
            except Exception as e:
                print(f"Warning: Could not analyze graph - {e}")
                continue

        results[name] = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "diameters": diameters,
            "connectivity_ratios": connectivity_ratios,
            "shortest_paths": shortest_paths,
        }
    return results


def plot_graph_properties_analysis(results, figsize=(18, 5)):
    _, axes = plt.subplots(1, 3, figsize=figsize)
    ax = axes[0]
    for i, (name, data) in enumerate(results.items()):
        color = get_dataset_color(name, i)
        if data["diameters"]:
            ax.hist(data["diameters"], bins=20, alpha=0.7, color=color, label=name)
    ax.set_xlabel("Graph Diameter")
    ax.set_ylabel("Frequency")
    ax.set_title("Graph Diameter Distribution")
    ax.legend()

    ax = axes[1]
    for i, (name, data) in enumerate(results.items()):
        color = get_dataset_color(name, i)
        if data["connectivity_ratios"]:
            ax.hist(data["connectivity_ratios"], bins=20, alpha=0.7, color=color, label=name)
    ax.set_xlabel("Graph Density")
    ax.set_ylabel("Frequency")
    ax.set_title("Graph Density Distribution")
    ax.legend()

    ax = axes[2]
    for i, (name, data) in enumerate(results.items()):
        color = get_dataset_color(name, i)
        if data["shortest_paths"]:
            ax.hist(data["shortest_paths"], bins=30, alpha=0.7, color=color, label=name)
    ax.set_xlabel("Shortest Path Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Shortest Path Length Distribution")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_graph_relationships(results, figsize=(18, 9)):
    _, axes = plt.subplots(2, 2, figsize=figsize)
    ax = axes[0, 0]
    for i, (name, data) in enumerate(results.items()):
        color = get_dataset_color(name, i)
        if data["diameters"] and data["num_nodes"]:
            ax.scatter(data["num_nodes"], data["diameters"], alpha=0.7, color=color, label=name)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Graph Diameter")
    ax.set_title("Graph Size vs Diameter")
    ax.legend()

    ax = axes[0, 1]
    for i, (name, data) in enumerate(results.items()):
        color = get_dataset_color(name, i)
        if data["connectivity_ratios"] and data["num_nodes"]:
            ax.scatter(data["num_nodes"], data["connectivity_ratios"], alpha=0.7, color=color, label=name)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Graph Density")
    ax.set_title("Graph Size vs Density")
    ax.legend()

    ax = axes[1, 0]
    for i, (name, data) in enumerate(results.items()):
        color = get_dataset_color(name, i)
        if data["num_edges"] and data["num_nodes"]:
            ax.scatter(data["num_nodes"], data["num_edges"], alpha=0.7, color=color, label=name)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Number of Edges")
    ax.set_title("Graph Size vs Edge Count")
    ax.legend()

    ax = axes[1, 1]
    for i, (name, data) in enumerate(results.items()):
        color = get_dataset_color(name, i)
        if data["connectivity_ratios"] and data["diameters"]:
            ax.scatter(data["connectivity_ratios"], data["diameters"], alpha=0.7, color=color, label=name)
    ax.set_xlabel("Graph Density")
    ax.set_ylabel("Graph Diameter")
    ax.set_title("Density vs Diameter")
    ax.legend()
    plt.tight_layout()
    plt.show()
