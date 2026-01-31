import time
import torch
import psutil
import os
from typing import Dict, Optional, Callable
from functools import wraps
from dataclasses import dataclass, field


@dataclass
class ComplexityStats:
    total_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    batch_times: list = field(default_factory=list)
    peak_memory_allocated: float = 0.0
    peak_memory_reserved: float = 0.0
    cpu_memory: float = 0.0
    num_parameters: int = 0
    num_trainable_parameters: int = 0
    flops: Optional[int] = None

    def to_dict(self) -> Dict:
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
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.stats = ComplexityStats()
        self.stats.num_parameters = sum(p.numel() for p in model.parameters())
        self.stats.num_trainable_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.reset_peak_memory_stats(device)

    def track(self, phase: str = "forward"):
        return _TrackingContext(self, phase)

    def update_memory_stats(self):
        if torch.cuda.is_available() and self.device != "cpu":
            self.stats.peak_memory_allocated = max(
                self.stats.peak_memory_allocated,
                torch.cuda.max_memory_allocated(self.device) / 1024**2
            )
            self.stats.peak_memory_reserved = max(
                self.stats.peak_memory_reserved,
                torch.cuda.max_memory_reserved(self.device) / 1024**2
            )
        process = psutil.Process(os.getpid())
        self.stats.cpu_memory = process.memory_info().rss / 1024**2

    def get_stats(self) -> ComplexityStats:
        self.update_memory_stats()
        return self.stats

    def reset(self):
        self.stats = ComplexityStats()
        self.stats.num_parameters = sum(p.numel() for p in self.model.parameters())
        self.stats.num_trainable_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        if torch.cuda.is_available() and self.device != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)


class _TrackingContext:
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
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


def measure_memory(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{func.__name__} peak GPU memory: {peak_mem:.2f} MB")
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024**2
        print(f"{func.__name__} CPU memory: {cpu_mem:.2f} MB")
        return result
    return wrapper


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_mb": total * 4 / 1024**2,
    }


def profile_model(
    model: torch.nn.Module,
    sample_data,
    device: str = "cpu",
    num_runs: int = 10
) -> Dict:
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(sample_data.to(device))
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
    if torch.cuda.is_available() and device != "cpu":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_mem = 0
    param_stats = count_parameters(model)
    return {
        "avg_forward_time_ms": sum(forward_times) / len(forward_times) * 1000,
        "std_forward_time_ms": torch.tensor(forward_times).std().item() * 1000,
        "peak_memory_mb": peak_mem,
        **param_stats,
    }
