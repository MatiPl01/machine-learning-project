"""
Trainer class for graph transformers.

Handles:
- Training loop with validation
- Checkpointing (as teacher requires!)
- Logging
- Early stopping
- Complexity tracking (memory, time)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import Dict, Optional
import os
import time
import json
from pathlib import Path

from .config import ExperimentConfig
from ..utils.metrics import compute_metrics, MetricTracker
from ..utils.complexity import ComplexityTracker


class Trainer:
    """
    Trainer for graph transformer models.
    
    Handles full training pipeline with logging, checkpointing, and complexity tracking.
    
    Example:
        >>> config = ExperimentConfig.load("configs/goat_molhiv.yaml")
        >>> trainer = Trainer(model, config, train_loader, val_loader)
        >>> trainer.train()
        >>> results = trainer.evaluate(test_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = config.training.device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Metric tracking
        self.metric_tracker = MetricTracker()
        
        # Complexity tracking (Teacher's requirement!)
        if config.training.track_complexity:
            self.complexity_tracker = ComplexityTracker(model, self.device)
        else:
            self.complexity_tracker = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = float('-inf') if config.training.eval_metric != 'mae' else float('inf')
        self.patience_counter = 0
        
        # Setup directories
        self._setup_directories()
        
        print(f"Trainer initialized for {config.experiment_name}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        cfg = self.config.training
        
        if cfg.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        cfg = self.config.training
        
        if cfg.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.num_epochs,
            )
        elif cfg.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif cfg.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on task"""
        model_cfg = self.config.model
        
        if model_cfg.out_channels == 1:
            # Binary classification or regression
            if self.config.training.eval_metric == "rocauc":
                return nn.BCEWithLogitsLoss()
            else:
                return nn.MSELoss()  # Regression
        else:
            # Multi-class classification
            return nn.CrossEntropyLoss(
                label_smoothing=self.config.training.label_smoothing
            )
    
    def _setup_directories(self):
        """Create directories for saving"""
        cfg = self.config.training
        
        # Checkpoints directory
        self.checkpoint_dir = Path(cfg.save_dir) / self.config.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logs directory
        self.log_dir = Path(cfg.log_dir) / self.config.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(str(self.checkpoint_dir / "config.yaml"))
    
    def train(self) -> Dict:
        """
        Main training loop.
        
        Returns:
            Dictionary with training history and final results
        """
        print(f"\nStarting training for {self.config.training.num_epochs} epochs...")
        print("=" * 80)
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            if self.val_loader is not None and epoch % self.config.training.eval_every == 0:
                val_metrics = self.evaluate(self.val_loader, split="val")
                
                # Check for improvement
                improved = self._check_improvement(val_metrics)
                
                # Save checkpoint
                if improved and self.config.training.save_best_only:
                    self.save_checkpoint("best_model.pt")
                
                # Early stopping
                if self.config.training.early_stop and not improved:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.training.patience:
                        print(f"\nEarly stopping at epoch {epoch}!")
                        break
                else:
                    self.patience_counter = 0
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log
            self._log_epoch(epoch, train_metrics, val_metrics if self.val_loader else None)
        
        # Final evaluation on test set
        test_metrics = None
        if self.test_loader is not None:
            print("\nEvaluating on test set...")
            self.load_checkpoint("best_model.pt")
            test_metrics = self.evaluate(self.test_loader, split="test")
            print(f"Test metrics: {test_metrics}")
        
        # Save final results
        results = {
            "train_history": self.metric_tracker.history,
            "test_metrics": test_metrics,
            "config": self.config,
        }
        
        self._save_results(results)
        
        return results
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        epoch_start_time = time.time()
        
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # Forward pass
            if self.complexity_tracker:
                with self.complexity_tracker.track("batch"):
                    self.optimizer.zero_grad()
                    out = self.model(data)
                    loss = self.criterion(out.squeeze(), data.y.float())
            else:
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out.squeeze(), data.y.float())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.append(out.detach().cpu())
            all_labels.append(data.y.detach().cpu())
            
            # Log batch
            if batch_idx % self.config.training.log_every == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # Compute epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.train_loader)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Determine task type for metrics
        task_type = "binary_classification" if self.config.training.eval_metric == "rocauc" else "regression"
        metrics = compute_metrics(all_labels, all_preds, task_type=task_type)
        metrics["loss"] = avg_loss
        metrics["time"] = epoch_time
        
        # Update tracker
        self.metric_tracker.update(self.current_epoch, metrics, split="train")
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict:
        """Evaluate on a dataset"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        eval_start_time = time.time()
        
        for data in loader:
            data = data.to(self.device)
            
            # Forward pass
            out = self.model(data)
            loss = self.criterion(out.squeeze(), data.y.float())
            
            total_loss += loss.item()
            all_preds.append(out.cpu())
            all_labels.append(data.y.cpu())
        
        eval_time = time.time() - eval_start_time
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        task_type = "binary_classification" if self.config.training.eval_metric == "rocauc" else "regression"
        metrics = compute_metrics(all_labels, all_preds, task_type=task_type)
        metrics["loss"] = total_loss / len(loader)
        metrics["time"] = eval_time
        
        # Update tracker
        self.metric_tracker.update(self.current_epoch, metrics, split=split)
        
        # Get complexity stats if tracking
        if self.complexity_tracker and split == "val":
            complexity_stats = self.complexity_tracker.get_stats()
            metrics.update(complexity_stats.to_dict())
        
        return metrics
    
    def _check_improvement(self, val_metrics: Dict) -> bool:
        """Check if validation metric improved"""
        metric_name = self.config.training.eval_metric
        current_metric = val_metrics.get(metric_name, 0.0)
        
        # For MAE, lower is better
        if metric_name == "mae":
            improved = current_metric < (self.best_val_metric - self.config.training.min_delta)
        else:
            # For accuracy/rocauc, higher is better
            improved = current_metric > (self.best_val_metric + self.config.training.min_delta)
        
        if improved:
            self.best_val_metric = current_metric
        
        return improved
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint {checkpoint_path} not found")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict]):
        """Log epoch results"""
        metric_name = self.config.training.eval_metric
        
        log_str = f"Epoch {epoch:3d} | "
        log_str += f"Train Loss: {train_metrics['loss']:.4f}, {metric_name}: {train_metrics.get(metric_name, 0):.4f} | "
        
        if val_metrics:
            log_str += f"Val Loss: {val_metrics['loss']:.4f}, {metric_name}: {val_metrics.get(metric_name, 0):.4f}"
        
        print(log_str)
    
    def _save_results(self, results: Dict):
        """Save final results to JSON"""
        results_path = self.log_dir / "results.json"
        
        # Convert to serializable format
        serializable_results = {
            "train_history": results["train_history"],
            "test_metrics": results["test_metrics"],
            "experiment_name": self.config.experiment_name,
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")


