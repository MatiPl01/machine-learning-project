"""
Utility functions for loading saved models.
"""

import torch
from pathlib import Path
from .models import get_model


def load_trained_model(
    checkpoint_path, model_class=None, input_dim=None, **model_kwargs
):
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the saved model checkpoint (.pt file)
        model_class: Model class or name (if None, will try to infer from checkpoint)
        input_dim: Input dimension (required if model_class is a string)
        **model_kwargs: Additional keyword arguments for model initialization

    Returns:
        Loaded model in evaluation mode
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model class
    if model_class is None:
        if "model_class" in checkpoint:
            model_class_name = checkpoint["model_class"]
            # Try to get from models module
            from .models import G2LFormer, GCNBaseline, GATBaseline

            model_classes = {
                "G2LFormer": G2LFormer,
                "GCNBaseline": GCNBaseline,
                "GATBaseline": GATBaseline,
            }
            if model_class_name in model_classes:
                model_class = model_classes[model_class_name]
            else:
                raise ValueError(f"Unknown model class: {model_class_name}")
        else:
            raise ValueError("model_class not specified and not found in checkpoint")

    # If model_class is a string, use get_model factory
    if isinstance(model_class, str):
        if input_dim is None:
            raise ValueError("input_dim required when model_class is a string")
        model = get_model(model_class, input_dim=input_dim, **model_kwargs)
    else:
        # model_class is a class, need input_dim and other params
        if input_dim is None:
            raise ValueError("input_dim required when model_class is a class")
        model = model_class(input_dim=input_dim, **model_kwargs)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def load_training_results(results_path):
    """
    Load training history and performance statistics from a results file.

    Args:
        results_path: Path to the saved results JSON file

    Returns:
        Dictionary containing training history and performance stats
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    import json

    with open(results_path, "r") as f:
        results = json.load(f)

    return results
