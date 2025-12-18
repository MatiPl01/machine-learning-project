"""
Complexity Analysis Tools

Track memory usage, training time, and computational complexity.
As per teacher's notes: "zrzucicie pamięci, czas trenowania"
"""

import time
import torch
import psutil
import os
from typing import Dict, Optional, Callable
from functools import wraps
from dataclasses import dataclass, field


@dataclass
class ComplexityStats:
    """Statistics for complexity analysis"""
    
    # Time statistics (in seconds)
    total_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    batch_times: list = field(default_factory=list)
    
    # Memory statistics (in MB)
    peak_memory_allocated: float = 0.0
    peak_memory_reserved: float = 0.0
    cpu_memory: float = 0.0
    
    # Model statistics
    num_parameters: int = 0
    num_trainable_parameters: int = 0
    
    # Computational statistics
    flops: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging"""
        return {
            "total_time": self.total_time,
            "avg_batch_time": sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0,
            "forward_time": self.forward_time,
            "backward_time": self.backward_time,
            "peak_memory_allocated_mb": self.peak_memory_allocated,
            "peak_memory_reserved_mb": self.peak_memory_reserved,
            "cpu_memory_mb": self.cpu_memory,
            "num_parameters": self.num_parameters,
            "num_trainable_parameters": self.num_trainable_parameters,
        }


class ComplexityTracker:
    """
    Track complexity metrics during training/inference.
    
    Usage:
        tracker = ComplexityTracker(model)
        
        with tracker.track("forward"):
            output = model(data)
        
        stats = tracker.get_stats()
        print(f"Peak memory: {stats.peak_memory_allocated:.2f} MB")
    """
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.stats = ComplexityStats()
        
        # Count parameters
        self.stats.num_parameters = sum(p.numel() for p in model.parameters())
        self.stats.num_trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Reset memory stats
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.reset_peak_memory_stats(device)
    
    def track(self, phase: str = "forward"):
        """
        Context manager for tracking a specific phase.
        
        Args:
            phase: "forward", "backward", or "both"
        """
        return _TrackingContext(self, phase)
    
    def update_memory_stats(self):
        """Update memory statistics"""
        if torch.cuda.is_available() and self.device != "cpu":
            # GPU memory
            self.stats.peak_memory_allocated = max(
                self.stats.peak_memory_allocated,
                torch.cuda.max_memory_allocated(self.device) / 1024**2  # Convert to MB
            )
            self.stats.peak_memory_reserved = max(
                self.stats.peak_memory_reserved,
                torch.cuda.max_memory_reserved(self.device) / 1024**2
            )
        
        # CPU memory
        process = psutil.Process(os.getpid())
        self.stats.cpu_memory = process.memory_info().rss / 1024**2  # MB
    
    def get_stats(self) -> ComplexityStats:
        """Get current complexity statistics"""
        self.update_memory_stats()
        return self.stats
    
    def reset(self):
        """Reset all statistics"""
        self.stats = ComplexityStats()
        self.stats.num_parameters = sum(p.numel() for p in self.model.parameters())
        self.stats.num_trainable_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)


class _TrackingContext:
    """Context manager for tracking a specific computation phase"""
    
    def __init__(self, tracker: ComplexityTracker, phase: str):
        self.tracker = tracker
        self.phase = phase
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        
        if self.phase == "forward":
            self.tracker.stats.forward_time += elapsed
        elif self.phase == "backward":
            self.tracker.stats.backward_time += elapsed
        elif self.phase == "batch":
            self.tracker.stats.batch_times.append(elapsed)
        
        self.tracker.stats.total_time += elapsed
        self.tracker.update_memory_stats()


def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.
    
    Example:
        @measure_time
        def train_epoch(model, loader):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


def measure_memory(func: Callable) -> Callable:
    """
    Decorator to measure peak memory usage of a function.
    
    Example:
        @measure_memory
        def forward_pass(model, data):
            return model(data)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Measure memory
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            print(f"{func.__name__} peak GPU memory: {peak_mem:.2f} MB")
        
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024**2
        print(f"{func.__name__} CPU memory: {cpu_mem:.2f} MB")
        
        return result
    return wrapper


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_mb": total * 4 / 1024**2,  # Assuming float32
    }


def profile_model(
    model: torch.nn.Module, 
    sample_data, 
    device: str = "cpu",
    num_runs: int = 10
) -> Dict:
    """
    Profile a model on sample data.
    
    Args:
        model: PyTorch model
        sample_data: Sample input data
        device: Device to run on
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with profiling results
    """
    model = model.to(device)
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(sample_data.to(device))
    
    # Profile forward pass
    forward_times = []
    
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(sample_data.to(device))
            if torch.cuda.is_available() and device != "cpu":
                torch.cuda.synchronize()
            forward_times.append(time.time() - start)
    
    # Memory stats
    if torch.cuda.is_available() and device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_mem = 0
    
    # Parameter count
    param_stats = count_parameters(model)
    
    return {
        "avg_forward_time_ms": sum(forward_times) / len(forward_times) * 1000,
        "std_forward_time_ms": torch.tensor(forward_times).std().item() * 1000,
        "peak_memory_mb": peak_mem,
        **param_stats,
    }


