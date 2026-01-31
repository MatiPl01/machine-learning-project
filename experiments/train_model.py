import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import argparse
import time
import json
from pathlib import Path

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models import (
    GOAT, Exphormer,
    GCN, GAT, GIN, GraphMLP,
    GCNVirtualNode, GINVirtualNode,
)
from src.utils.data import (
    load_zinc_dataset,
    load_molhiv_dataset,
    load_molpcba_dataset,
    load_ppa_dataset,
    load_peptides_func_dataset,
)
from src.utils.positional_encodings import precompute_positional_encodings
from src.utils.complexity import count_parameters

MODEL_REGISTRY = {
    'goat': GOAT,
    'exphormer': Exphormer,
    'gcn': GCN,
    'gat': GAT,
    'gin': GIN,
    'graphmlp': GraphMLP,
    'gcn_virtualnode': GCNVirtualNode,
    'gcn_vn': GCNVirtualNode,
    'gin_virtualnode': GINVirtualNode,
    'gin_vn': GINVirtualNode,
}

DATASET_REGISTRY = {
    'zinc': (load_zinc_dataset, 'regression', 'mae', 1),
    'molhiv': (load_molhiv_dataset, 'binary', 'rocauc', 1),
    'molpcba': (load_molpcba_dataset, 'multi_label', 'ap', 128),
    'ppa': (load_ppa_dataset, 'multi_class', 'accuracy', 37),
    'peptides': (load_peptides_func_dataset, 'multi_label', 'ap', 10),
}


def get_config(args):
    if args.mode == 'cpu':
        config = {
            'use_subset': True,
            'subset_size': 500,
            'batch_size': 32,
            'num_epochs': 10,
            'hidden_dim': 64,
            'num_layers': 3,
            'num_heads': 4,
            'lr': 1e-3,
            'dropout': 0.1,
            'pe_dim': 8,
            'device': 'cpu',
        }
    else:
        config = {
            'use_subset': False,
            'subset_size': None,
            'batch_size': 64,
            'num_epochs': 100,
            'hidden_dim': 256,
            'num_layers': 5,
            'num_heads': 8,
            'lr': 1e-4,
            'dropout': 0.1,
            'pe_dim': 16,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['lr'] = args.lr
    if args.hidden_dim:
        config['hidden_dim'] = args.hidden_dim
    if args.device:
        config['device'] = args.device
    return config


def load_dataset(dataset_name, config):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    load_fn, task_type, metric, out_channels = DATASET_REGISTRY[dataset_name]
    print(f"Loading dataset: {dataset_name}")
    dataset, split_idx = load_fn()
    if config['use_subset']:
        print(f"Using subset: {config['subset_size']} graphs")
        indices = torch.randperm(len(dataset))[:config['subset_size']]
        train_size = int(0.8 * config['subset_size'])
        val_size = int(0.1 * config['subset_size'])
        split_idx = {
            'train': indices[:train_size],
            'valid': indices[train_size:train_size+val_size],
            'test': indices[train_size+val_size:],
        }
    print("Computing positional encodings...")
    dataset = precompute_positional_encodings(dataset, pe_type='laplacian', pe_dim=config['pe_dim'])
    sample = dataset[0]
    in_channels = sample.x.shape[1] if sample.x.dim() > 1 else 1
    dataset_info = {
        'task_type': task_type,
        'metric': metric,
        'in_channels': in_channels,
        'out_channels': out_channels,
    }
    print(f"Dataset loaded: {len(dataset)} graphs")
    print(f"  Train: {len(split_idx['train'])}, Val: {len(split_idx['valid'])}, Test: {len(split_idx['test'])}")
    return dataset, split_idx, dataset_info


def create_model(model_name, config, dataset_info):
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    model_cls = MODEL_REGISTRY[model_name]
    kwargs = {
        'in_channels': dataset_info['in_channels'],
        'hidden_channels': config['hidden_dim'],
        'out_channels': dataset_info['out_channels'],
        'num_layers': config['num_layers'],
        'dropout': config['dropout'],
    }
    if model_name in ['goat', 'exphormer']:
        kwargs['num_heads'] = config['num_heads']
        kwargs['pe_dim'] = config['pe_dim']
    elif model_name in ['gat']:
        kwargs['num_heads'] = config['num_heads']
    elif model_name in ['gin', 'gin_virtualnode', 'gin_vn']:
        kwargs['train_eps'] = True
    return model_cls(**kwargs)


def train_epoch(model, loader, optimizer, device, task_type):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        if task_type == 'regression':
            y = batch.y.float().view(-1, 1)
            loss = F.mse_loss(out, y)
        elif task_type == 'binary':
            y = batch.y.float().view(-1, 1)
            loss = F.binary_cross_entropy_with_logits(out, y)
        elif task_type == 'multi_label':
            y = batch.y.float()
            mask = ~torch.isnan(y)
            if mask.sum() > 0:
                loss = F.binary_cross_entropy_with_logits(out[mask], y[mask])
            else:
                loss = torch.tensor(0.0, device=device)
        elif task_type == 'multi_class':
            y = batch.y.view(-1)
            loss = F.cross_entropy(out, y)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, task_type, metric):
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        if task_type == 'regression':
            y = batch.y.float().view(-1, 1)
            all_preds.append(out.cpu())
            all_labels.append(y.cpu())
        elif task_type == 'binary':
            y = batch.y.float().view(-1, 1)
            all_preds.append(torch.sigmoid(out).cpu())
            all_labels.append(y.cpu())
        elif task_type == 'multi_label':
            y = batch.y.float()
            all_preds.append(torch.sigmoid(out).cpu())
            all_labels.append(y.cpu())
        elif task_type == 'multi_class':
            y = batch.y.view(-1)
            all_preds.append(out.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if metric == 'mae':
        score = F.l1_loss(all_preds, all_labels).item()
    elif metric == 'rocauc':
        from sklearn.metrics import roc_auc_score
        try:
            score = roc_auc_score(all_labels.numpy(), all_preds.numpy())
        except Exception:
            score = 0.5
    elif metric == 'ap':
        from sklearn.metrics import average_precision_score
        mask = ~torch.isnan(all_labels)
        try:
            score = average_precision_score(all_labels[mask].numpy(), all_preds[mask].numpy())
        except Exception:
            score = 0.0
    elif metric == 'accuracy':
        preds = all_preds.argmax(dim=1)
        score = (preds == all_labels).float().mean().item()
    else:
        score = 0.0
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = get_config(args)
    device = torch.device(config['device'])

    print("="*70)
    print(f"TRAINING: {args.model.upper()} on {args.dataset.upper()}")
    print(f"Mode: {args.mode.upper()}, Device: {device}")
    print("="*70)

    dataset, split_idx, dataset_info = load_dataset(args.dataset, config)
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset[split_idx['valid']], batch_size=config['batch_size'])
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=config['batch_size'])

    model = create_model(args.model, config, dataset_info)
    model = model.to(device)
    num_params = count_parameters(model)['total']
    print(f"\nModel: {args.model}")
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    task_type = dataset_info['task_type']
    metric = dataset_info['metric']
    higher_is_better = metric in ['rocauc', 'ap', 'accuracy']
    best_val_score = float('-inf') if higher_is_better else float('inf')
    start_time = time.time()

    print(f"\nTraining for {config['num_epochs']} epochs...")
    print(f"Task: {task_type}, Metric: {metric}")

    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device, task_type)
        val_score = evaluate(model, val_loader, device, task_type, metric)
        scheduler.step()
        if higher_is_better:
            if val_score > best_val_score:
                best_val_score = val_score
        else:
            if val_score < best_val_score:
                best_val_score = val_score
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config['num_epochs']}: "
                  f"Loss={train_loss:.4f}, Val {metric}={val_score:.4f} (best={best_val_score:.4f})")

    train_time = time.time() - start_time
    test_score = evaluate(model, test_loader, device, task_type, metric)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best Val {metric}: {best_val_score:.4f}")
    print(f"Test {metric}: {test_score:.4f}")
    print(f"Training time: {train_time:.1f}s")
    print(f"Parameters: {num_params:,}")

    if args.save_results:
        results = {
            'model': args.model,
            'dataset': args.dataset,
            'mode': args.mode,
            'config': config,
            'dataset_info': dataset_info,
            'best_val_score': best_val_score,
            'test_score': test_score,
            'train_time': train_time,
            'num_params': num_params,
        }
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{args.model}_{args.dataset}_{args.mode}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
