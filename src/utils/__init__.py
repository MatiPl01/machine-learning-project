"""
Utility modules for Graph Transformers project.

This package contains:
- data: Data loading and analysis utilities
- analysis: Visualization and analysis functions
- constants: Configuration and constants
- complexity: Memory and time tracking
- metrics: Evaluation metrics (ROC-AUC, MAE, etc.)
- positional_encodings: Laplacian, random walk, degree encodings
"""

# Only import the main public API functions
from .data import (
    load_molhiv_dataset,
    load_peptides_func_dataset,
    load_zinc_dataset,
    print_dataset_summary,
)

from .analysis import (
    visualize_sample_graphs,
    plot_graph_size_distributions,
)

# Constants that users might need
from .constants import (
    DATASET_COLORS,
    DISTINCT_COLORS,
)

# Complexity tracking (Teacher's requirement!)
from .complexity import (
    ComplexityTracker,
    count_parameters,
)

# Metrics
from .metrics import (
    compute_metrics,
    MetricTracker,
)

# Positional encodings
from .positional_encodings import (
    precompute_positional_encodings,
)

__all__ = [
    # Main data loading functions
    "load_molhiv_dataset",
    "load_peptides_func_dataset",
    "load_zinc_dataset",
    "print_dataset_summary",
    # Main visualization functions
    "visualize_sample_graphs",
    "plot_graph_size_distributions",
    # Useful constants
    "DATASET_COLORS",
    "DISTINCT_COLORS",
    # Complexity tracking
    "ComplexityTracker",
    "count_parameters",
    # Metrics
    "compute_metrics",
    "MetricTracker",
    # Positional encodings
    "precompute_positional_encodings",
]
