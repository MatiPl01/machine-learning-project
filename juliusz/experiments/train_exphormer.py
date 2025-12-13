"""
Training script for Exphormer model.

Usage:
    python experiments/train_exphormer.py --config configs/exphormer_molhiv.yaml
    python experiments/train_exphormer.py --config configs/exphormer_zinc.yaml
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import argparse

from juliusz.models.exphormer import Exphormer
from juliusz.training import Trainer, ExperimentConfig
from juliusz.utils.positional_encodings import precompute_positional_encodings
from src.utils.data import load_molhiv_dataset, load_zinc_dataset, load_peptides_func_dataset

from torch_geometric.loader import DataLoader


def load_dataset(config: ExperimentConfig):
    """Load dataset based on config"""
    dataset_name = config.training.dataset_name
    data_root = config.training.data_root
    
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "ogbg-molhiv":
        dataset, split_idx = load_molhiv_dataset(data_root)
    elif dataset_name == "zinc":
        dataset, split_idx = load_zinc_dataset(data_root)
    elif dataset_name == "peptides-func":
        dataset, split_idx = load_peptides_func_dataset(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Precompute positional encodings
    print("Precomputing positional encodings...")
    dataset = precompute_positional_encodings(
        dataset,
        pe_type=config.model.pe_type,
        pe_dim=config.model.pe_dim,
    )
    
    return dataset, split_idx


def create_dataloaders(dataset, split_idx, config: ExperimentConfig):
    """Create train/val/test dataloaders"""
    batch_size = config.training.batch_size
    num_workers = config.training.num_workers
    
    train_loader = DataLoader(
        dataset[split_idx['train']],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        dataset[split_idx['valid']],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        dataset[split_idx['test']],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


def create_model(config: ExperimentConfig) -> Exphormer:
    """Create Exphormer model from config"""
    model_cfg = config.model
    
    model = Exphormer(
        in_channels=model_cfg.in_channels,
        hidden_channels=model_cfg.hidden_channels,
        out_channels=model_cfg.out_channels,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        expander_degree=model_cfg.expander_degree,
        expander_method=model_cfg.expander_method,
        pe_dim=model_cfg.pe_dim,
        dropout=model_cfg.dropout,
        task_type=model_cfg.task_type,
        pooling_type=model_cfg.pooling_type,
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Exphormer model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    config = ExperimentConfig.load(args.config)
    
    # Override device if specified
    if args.device:
        config.training.device = args.device
    
    # Override seed if specified
    if args.seed:
        config.training.seed = args.seed
    
    # Set random seed
    torch.manual_seed(config.training.seed)
    
    print("=" * 80)
    print(config)
    print("=" * 80)
    
    # Load dataset
    dataset, split_idx = load_dataset(config)
    train_loader, val_loader, test_loader = create_dataloaders(dataset, split_idx, config)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_loader.dataset):,} graphs")
    print(f"  Val:   {len(val_loader.dataset):,} graphs")
    print(f"  Test:  {len(test_loader.dataset):,} graphs")
    
    # Create model
    model = create_model(config)
    print(f"\nModel created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print complexity info
    complexity_stats = model.get_complexity_stats()
    print(f"\nComplexity statistics:")
    for key, value in complexity_stats.items():
        print(f"  {key}: {value}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    
    # Train
    results = trainer.train()
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation {config.training.eval_metric}: {trainer.best_val_metric:.4f}")
    if results['test_metrics']:
        print(f"Test {config.training.eval_metric}: {results['test_metrics'].get(config.training.eval_metric, 0):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()


