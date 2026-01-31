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
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    metrics = {}

    if task_type == "binary_classification":
        if y_pred.ndim > 1 and y_pred.shape[1] == 2:
            y_pred_proba = torch.softmax(torch.from_numpy(y_pred), dim=1)[:, 1].numpy()
        elif y_pred.ndim > 1 and y_pred.shape[1] == 1:
            y_pred_proba = torch.sigmoid(torch.from_numpy(y_pred)).squeeze().numpy()
        else:
            y_pred_proba = y_pred.squeeze()
        try:
            metrics["rocauc"] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics["rocauc"] = 0.0
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)

    elif task_type == "classification":
        y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
        metrics["accuracy"] = accuracy_score(y_true, y_pred_class)
        if y_pred.ndim > 1 and num_classes is not None:
            try:
                y_pred_proba = torch.softmax(torch.from_numpy(y_pred), dim=1).numpy()
                metrics["rocauc"] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='macro'
                )
            except ValueError:
                pass

    elif task_type == "regression":
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(np.mean((y_true - y_pred) ** 2))
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return metrics


class MetricTracker:
    def __init__(self):
        self.history = {"train": {}, "val": {}, "test": {}}

    def update(self, epoch: int, metrics: Dict[str, float], split: str = "train"):
        if split not in self.history:
            self.history[split] = {}
        for key, value in metrics.items():
            if key not in self.history[split]:
                self.history[split][key] = []
            self.history[split][key].append((epoch, value))

    def get_best(self, metric: str = "rocauc", split: str = "val") -> tuple:
        if split not in self.history or metric not in self.history[split]:
            return None, None
        values = self.history[split][metric]
        if metric in ["loss", "mae", "rmse"]:
            best_idx = np.argmin([v for _, v in values])
        else:
            best_idx = np.argmax([v for _, v in values])
        return values[best_idx]

    def plot(self, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        all_metrics = set()
        for split_data in self.history.values():
            all_metrics.update(split_data.keys())
        num_metrics = len(all_metrics)
        if num_metrics == 0:
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
        else:
            plt.show()

    def save(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, path: str):
        import json
        with open(path, 'r') as f:
            self.history = json.load(f)
