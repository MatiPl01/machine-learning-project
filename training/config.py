"""
Configuration classes for training experiments.

Uses dataclasses for easy configuration management.
Can be saved/loaded from YAML files.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import yaml
import os


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    
    # Model type
    model_type: str = "goat"  # 'goat' or 'exphormer'
    
    # Architecture
    in_channels: int = 32
    hidden_channels: int = 256
    out_channels: int = 2
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    
    # Positional encodings
    pe_type: str = "laplacian"  # 'laplacian', 'random_walk', 'degree', 'none'
    pe_dim: int = 8
    
    # Task-specific
    task_type: str = "graph_classification"  # or 'node_classification'
    pooling_type: str = "mean"  # 'mean', 'max', 'add'
    
    # Model-specific parameters
    # For GOAT:
    num_virtual_nodes: int = 1
    
    # For Exphormer:
    expander_degree: int = 4
    expander_method: str = "random"  # 'random' or 'circulant'


@dataclass
class TrainingConfig:
    """Configuration for training"""
    
    # Dataset
    dataset_name: str = "ogbg-molhiv"  # 'ogbg-molhiv', 'zinc', 'peptides-func'
    data_root: str = "./data"
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimization
    optimizer: str = "adam"  # 'adam', 'adamw', 'sgd'
    scheduler: str = "cosine"  # 'cosine', 'step', 'none'
    warmup_epochs: int = 5
    
    # Regularization
    gradient_clip: Optional[float] = 1.0
    label_smoothing: float = 0.0
    
    # Early stopping
    early_stop: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_best_only: bool = True
    
    # Logging
    log_dir: str = "./logs"
    log_every: int = 10  # Log every N batches
    
    # Evaluation
    eval_metric: str = "rocauc"  # 'rocauc', 'accuracy', 'mae'
    eval_every: int = 1  # Evaluate every N epochs
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = "cpu"  # 'cpu', 'cuda', 'cuda:0', etc.
    num_workers: int = 0  # Number of dataloader workers
    
    # Complexity tracking (Teacher's requirement!)
    track_complexity: bool = True
    profile_memory: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "goat_molhiv"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct nested dataclasses
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            model=model_config,
            training=training_config,
            experiment_name=config_dict.get('experiment_name', 'experiment'),
            description=config_dict.get('description', ''),
            tags=config_dict.get('tags', []),
        )
    
    def __str__(self) -> str:
        """Pretty print configuration"""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Description: {self.description}",
            "",
            "Model Configuration:",
        ]
        for key, value in asdict(self.model).items():
            lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append("Training Configuration:")
        for key, value in asdict(self.training).items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


def create_default_configs() -> dict:
    """
    Create default configurations for all experiments.
    
    Returns:
        Dictionary of {experiment_name: config}
    """
    configs = {}
    
    # GOAT on MolHIV
    configs["goat_molhiv"] = ExperimentConfig(
        experiment_name="goat_molhiv",
        description="GOAT model on OGB-MolHIV dataset",
        tags=["goat", "molhiv", "binary_classification"],
        model=ModelConfig(
            model_type="goat",
            in_channels=9,  # MolHIV has 9 node features
            hidden_channels=256,
            out_channels=1,  # Binary classification
            num_layers=4,
            num_heads=8,
            pe_dim=8,
            task_type="graph_classification",
        ),
        training=TrainingConfig(
            dataset_name="ogbg-molhiv",
            batch_size=32,
            num_epochs=100,
            learning_rate=1e-4,
            eval_metric="rocauc",
        ),
    )
    
    # Exphormer on MolHIV
    configs["exphormer_molhiv"] = ExperimentConfig(
        experiment_name="exphormer_molhiv",
        description="Exphormer model on OGB-MolHIV dataset",
        tags=["exphormer", "molhiv", "binary_classification"],
        model=ModelConfig(
            model_type="exphormer",
            in_channels=9,
            hidden_channels=256,
            out_channels=1,
            num_layers=4,
            num_heads=8,
            pe_dim=8,
            expander_degree=4,
            task_type="graph_classification",
        ),
        training=TrainingConfig(
            dataset_name="ogbg-molhiv",
            batch_size=32,
            num_epochs=100,
            learning_rate=1e-4,
            eval_metric="rocauc",
        ),
    )
    
    # GOAT on ZINC
    configs["goat_zinc"] = ExperimentConfig(
        experiment_name="goat_zinc",
        description="GOAT model on ZINC dataset",
        tags=["goat", "zinc", "regression"],
        model=ModelConfig(
            model_type="goat",
            in_channels=28,  # ZINC node features
            hidden_channels=128,
            out_channels=1,  # Regression
            num_layers=4,
            num_heads=4,
            pe_dim=8,
            task_type="graph_classification",
        ),
        training=TrainingConfig(
            dataset_name="zinc",
            batch_size=64,
            num_epochs=200,
            learning_rate=5e-4,
            eval_metric="mae",
        ),
    )
    
    # Exphormer on ZINC
    configs["exphormer_zinc"] = ExperimentConfig(
        experiment_name="exphormer_zinc",
        description="Exphormer model on ZINC dataset",
        tags=["exphormer", "zinc", "regression"],
        model=ModelConfig(
            model_type="exphormer",
            in_channels=28,
            hidden_channels=128,
            out_channels=1,
            num_layers=4,
            num_heads=4,
            pe_dim=8,
            expander_degree=4,
            task_type="graph_classification",
        ),
        training=TrainingConfig(
            dataset_name="zinc",
            batch_size=64,
            num_epochs=200,
            learning_rate=5e-4,
            eval_metric="mae",
        ),
    )
    
    return configs


if __name__ == "__main__":
    # Example: Create and save default configs
    configs = create_default_configs()
    
    for name, config in configs.items():
        config.save(f"./configs/{name}.yaml")
    
    print("Default configurations created!")

