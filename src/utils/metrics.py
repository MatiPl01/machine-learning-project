"""
Evaluation Metrics for Graph Classification/Regression

As per teacher's notes: "AU-ROC albo ACCURACY"
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from typing import Dict, Optional


def compute_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    task_type: str = "classification",
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics based on task type.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted values (logits for classification, values for regression)
        task_type: 'classification', 'binary_classification', or 'regression'
        num_classes: Number of classes (for multi-class classification)
        
    Returns:
        Dictionary of metrics
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    metrics = {}
    
    if task_type == "binary_classification":
        # Binary classification: use ROC-AUC (teacher's requirement!)
        # y_pred should be probabilities or logits
        if y_pred.ndim > 1 and y_pred.shape[1] == 2:
            # Two-class logits - use second class probability
            y_pred_proba = torch.softmax(torch.from_numpy(y_pred), dim=1)[:, 1].numpy()
        elif y_pred.ndim > 1 and y_pred.shape[1] == 1:
            # Single output - apply sigmoid
            y_pred_proba = torch.sigmoid(torch.from_numpy(y_pred)).squeeze().numpy()
        else:
            # Already probabilities
            y_pred_proba = y_pred.squeeze()
        
        # ROC-AUC (main metric for MolHIV)
        try:
            metrics["rocauc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
            metrics["rocauc"] = 0.0
        
        # Accuracy
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        
    elif task_type == "classification":
        # Multi-class classification
        y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        
        # Multi-class ROC-AUC (if probabilities provided)
        if y_pred.ndim > 1 and num_classes is not None:
            try:
                y_pred_proba = torch.softmax(torch.from_numpy(y_pred), dim=1).numpy()
                metrics["rocauc"] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='macro'
                )
            except ValueError:
                pass
    
    elif task_type == "regression":
        # Regression: use MAE as primary metric
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    return metrics


class MetricTracker:
    """
    Track metrics across epochs.
    
    Usage:
        tracker = MetricTracker()
        
        for epoch in range(num_epochs):
            train_metrics = evaluate(model, train_loader)
            val_metrics = evaluate(model, val_loader)
            
            tracker.update(epoch, train_metrics, val_metrics)
            
        tracker.plot()
    """
    
    def __init__(self):
        self.history = {
            "train": {},
            "val": {},
            "test": {}
        }
    
    def update(self, epoch: int, metrics: Dict[str, float], split: str = "train"):
        """Update metrics for a given epoch and split"""
        if split not in self.history:
            self.history[split] = {}
        
        for key, value in metrics.items():
            if key not in self.history[split]:
                self.history[split][key] = []
            self.history[split][key].append((epoch, value))
    
    def get_best(self, metric: str = "rocauc", split: str = "val") -> tuple:
        """
        Get best epoch and value for a metric.
        
        Returns:
            (best_epoch, best_value)
        """
        if split not in self.history or metric not in self.history[split]:
            return None, None
        
        values = self.history[split][metric]
        
        # For loss/mae, lower is better; for accuracy/rocauc, higher is better
        if metric in ["loss", "mae", "rmse"]:
            best_idx = np.argmin([v for _, v in values])
        else:
            best_idx = np.argmax([v for _, v in values])
        
        return values[best_idx]
    
    def plot(self, save_path: Optional[str] = None):
        """Plot metric history"""
        import matplotlib.pyplot as plt
        
        # Determine number of unique metrics
        all_metrics = set()
        for split_data in self.history.values():
            all_metrics.update(split_data.keys())
        
        num_metrics = len(all_metrics)
        if num_metrics == 0:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 4))
        if num_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, sorted(all_metrics)):
            for split in ["train", "val", "test"]:
                if split in self.history and metric in self.history[split]:
                    epochs, values = zip(*self.history[split][metric])
                    ax.plot(epochs, values, label=split, marker='o')
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} over Epochs")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def save(self, path: str):
        """Save history to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: str):
        """Load history from file"""
        import json
        with open(path, 'r') as f:
            self.history = json.load(f)


