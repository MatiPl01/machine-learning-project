"""
Training utilities for Graph Transformers project.
Includes training loops, evaluation metrics, and performance tracking.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score
import warnings
import json
from pathlib import Path

warnings.filterwarnings("ignore")


class PerformanceTracker:
    """Track training time, memory usage, and other metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_times = []
        self.memory_usage = []
        self.peak_memory = 0

    def start_timer(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def end_timer(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        self.train_times.append(elapsed)

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
            self.peak_memory = max(self.peak_memory, peak_mem)
            self.memory_usage.append(peak_mem)
        else:
            # CPU memory tracking (approximate)
            try:
                import psutil

                process = psutil.Process()
                mem_gb = process.memory_info().rss / 1e9
                self.memory_usage.append(mem_gb)
                self.peak_memory = max(self.peak_memory, mem_gb)
            except ImportError:
                # psutil not available, skip memory tracking
                self.memory_usage.append(0)

        return elapsed

    def get_stats(self):
        return {
            "avg_train_time": np.mean(self.train_times) if self.train_times else 0,
            "total_train_time": sum(self.train_times),
            "peak_memory_gb": self.peak_memory,
            "avg_memory_gb": np.mean(self.memory_usage) if self.memory_usage else 0,
        }


def compute_metrics(y_pred, y_true, task_type="classification"):
    """Compute task-specific metrics."""
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    if task_type == "classification":
        # Binary classification
        if y_pred_np.shape[1] == 1:
            # Check if predictions are already probabilities (in [0,1]) or logits
            # If values are outside [0,1], they're logits and need sigmoid
            if y_pred_np.min() < 0 or y_pred_np.max() > 1:
                # Logits - apply sigmoid
                y_pred_proba = torch.sigmoid(torch.from_numpy(y_pred_np)).numpy()
            else:
                # Already probabilities
                y_pred_proba = y_pred_np

            y_pred_binary = (y_pred_proba > 0.5).astype(int).flatten()
            y_true_binary = y_true_np.flatten()

            try:
                roc_auc = roc_auc_score(y_true_binary, y_pred_proba.flatten())
            except ValueError:
                roc_auc = 0.0

            acc = accuracy_score(y_true_binary, y_pred_binary)
            return {"roc_auc": roc_auc, "accuracy": acc}
        else:
            # Multi-class
            y_pred_class = np.argmax(y_pred_np, axis=1)
            acc = accuracy_score(y_true_np, y_pred_class)
            return {"accuracy": acc}

    elif task_type == "regression":
        mae = mean_absolute_error(y_true_np, y_pred_np)
        mse = np.mean((y_pred_np - y_true_np) ** 2)
        rmse = np.sqrt(mse)
        return {"mae": mae, "mse": mse, "rmse": rmse}

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def train_epoch(
    model, loader, optimizer, criterion, device, task_type="classification"
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)

        # Get labels
        if hasattr(batch, "y"):
            y = batch.y
            if y.dim() > 1 and y.size(1) == 1:
                y = y.squeeze(1)
        else:
            continue

        # Compute loss
        if task_type == "classification":
            if out.size(1) == 1:
                loss = criterion(out.squeeze(), y.float())
            else:
                loss = criterion(out, y.long())
        else:  # regression
            loss = criterion(out.squeeze(), y.float())

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(out.detach())
        all_labels.append(y.detach())

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_preds, all_labels, task_type)
    metrics["loss"] = total_loss / len(loader)

    return metrics


def evaluate(model, loader, criterion, device, task_type="classification"):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            out = model(batch.x, batch.edge_index, batch.batch)

            # Get labels
            if hasattr(batch, "y"):
                y = batch.y
                if y.dim() > 1 and y.size(1) == 1:
                    y = y.squeeze(1)
            else:
                continue

            # Compute loss
            if task_type == "classification":
                if out.size(1) == 1:
                    loss = criterion(out.squeeze(), y.float())
                else:
                    loss = criterion(out, y.long())
            else:  # regression
                loss = criterion(out.squeeze(), y.float())

            total_loss += loss.item()
            all_preds.append(out.detach())
            all_labels.append(y.detach())

    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_preds, all_labels, task_type)
    metrics["loss"] = total_loss / len(loader)

    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    device,
    task_type="classification",
    lr=0.001,
    weight_decay=0.0001,
    max_epochs=None,
    verbose=True,
    save_dir=None,
    model_name=None,
    patience=10,
    min_delta=0.0,
):
    """
    Train a model with limited epochs for initial results.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs (can be limited)
        device: Device to train on
        task_type: 'classification' or 'regression'
        lr: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum epochs to run (for quick testing)
        verbose: Print progress
        save_dir: Directory to save model checkpoints (default: './checkpoints')
        model_name: Name for saved model files (default: model class name)
        patience: Early stopping patience (epochs without improvement). 0 to disable.
        min_delta: Minimum change to qualify as improvement
    """
    if max_epochs is not None:
        num_epochs = min(num_epochs, max_epochs)

    # Setup save directory
    if save_dir is None:
        save_dir = "./checkpoints"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = model.__class__.__name__

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task_type == "classification":
        # Check output dimension to determine loss function
        dummy_batch = next(iter(train_loader))
        with torch.no_grad():
            dummy_out = model(
                dummy_batch.x.to(device),
                dummy_batch.edge_index.to(device),
                dummy_batch.batch.to(device),
            )
        if dummy_out.size(1) == 1:
            criterion = nn.BCEWithLogitsLoss()  # Binary classification with logits
        else:
            criterion = nn.CrossEntropyLoss()  # Multi-class
    else:
        criterion = nn.MSELoss()

    # Performance tracking
    tracker = PerformanceTracker()

    # Training history
    history = {"train_loss": [], "train_metrics": [], "val_loss": [], "val_metrics": []}

    best_val_metric = float("-inf")
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model.__class__.__name__}")
        print(f"Task: {task_type}, Epochs: {num_epochs}, Device: {device}")
        if patience > 0:
            print(f"Early stopping: patience={patience}, min_delta={min_delta}")
        print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        tracker.start_timer()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, task_type
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, task_type)

        elapsed = tracker.end_timer()

        # Store history
        history["train_loss"].append(train_metrics["loss"])
        history["train_metrics"].append(
            {k: v for k, v in train_metrics.items() if k != "loss"}
        )
        history["val_loss"].append(val_metrics["loss"])
        history["val_metrics"].append(
            {k: v for k, v in val_metrics.items() if k != "loss"}
        )

        # Track best model
        if task_type == "classification":
            metric_key = "roc_auc" if "roc_auc" in val_metrics else "accuracy"
            val_metric = val_metrics[metric_key]
        else:
            metric_key = "mae"
            # For regression, lower is better, so negate
            val_metric = -val_metrics[metric_key]

        # Check for improvement
        improved = val_metric > (best_val_metric + min_delta)

        if improved:
            best_val_metric = val_metric
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.2f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, ", end="")
            for k, v in train_metrics.items():
                if k != "loss":
                    print(f"{k}: {v:.4f}", end=" ")
            print()
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, ", end="")
            for k, v in val_metrics.items():
                if k != "loss":
                    print(f"{k}: {v:.4f}", end=" ")
            if improved:
                print(" [best]", end="")
            print()

        # Early stopping check
        if patience > 0 and patience_counter >= patience:
            early_stopped = True
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(
                    f"No improvement for {patience} epochs (best at epoch {best_epoch})"
                )
            break

    # Get performance stats
    perf_stats = tracker.get_stats()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

        # Save best model to disk
        model_path = save_dir / f"{model_name}_best.pt"
        torch.save(
            {
                "model_state_dict": best_model_state,
                "model_class": model.__class__.__name__,
                "best_val_metric": best_val_metric,
                "num_epochs": num_epochs,
                "task_type": task_type,
                "hyperparameters": {
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
            },
            model_path,
        )

        # Save training history and performance stats
        results_path = save_dir / f"{model_name}_results.json"
        results = {
            "history": {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "train_metrics": history["train_metrics"],
                "val_metrics": history["val_metrics"],
            },
            "performance": perf_stats,
            "best_val_metric": best_val_metric,
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if verbose:
            print(f"Model saved to: {model_path}")
            print(f"Results saved to: {results_path}")

    if verbose:
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Val Metric: {best_val_metric:.4f} (at epoch {best_epoch})")
        print(f"Total Epochs: {epoch+1}/{num_epochs}")
        if early_stopped:
            print("Early stopping triggered")
        print(f"Avg Time/Epoch: {perf_stats['avg_train_time']:.2f}s")
        print(f"Peak Memory: {perf_stats['peak_memory_gb']:.2f} GB")
        print(f"{'='*60}\n")

    return history, perf_stats


def get_task_type(dataset_name):
    """Determine task type from dataset name."""
    if "molhiv" in dataset_name.lower():
        return "classification"
    elif "zinc" in dataset_name.lower():
        return "regression"
    elif "peptides" in dataset_name.lower():
        return "regression"
    else:
        return "classification"  # default
