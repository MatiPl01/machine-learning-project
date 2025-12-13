# Person A (Juliusz) - Graph Transformers Implementation

This directory contains the implementation of GOAT and Exphormer models with full training infrastructure.

## 🎯 What's Implemented

### ✅ Models
- **GOAT**: Global attention transformer with O(N) complexity
- **Exphormer**: Sparse transformer using expander graphs

### ✅ Utilities
- **Positional Encodings**: Laplacian, Random Walk, Degree centrality
- **Complexity Tracking**: Memory usage, training time (as per teacher's requirements!)
- **Metrics**: ROC-AUC, MAE, accuracy

### ✅ Training Infrastructure
- Full training loop with validation
- Checkpointing (best model saving)
- Early stopping with patience
- Learning rate scheduling
- Complexity analysis during training

### ✅ Configuration System
- YAML-based experiment configs
- Easy hyperparameter management
- Reproducibility support

## 📁 Directory Structure

```
juliusz/
├── models/                    # Model implementations
│   ├── goat.py               # GOAT transformer
│   ├── exphormer.py          # Exphormer transformer
│   ├── base.py               # Base transformer class
│   └── layers.py             # Common layers
├── training/                  # Training infrastructure
│   ├── trainer.py            # Main trainer class
│   ├── config.py             # Configuration classes
│   └── utils.py              # Training utilities
├── utils/                     # Utilities
│   ├── positional_encodings.py
│   ├── complexity.py         # Memory/time tracking
│   └── metrics.py            # Evaluation metrics
├── configs/                   # Experiment configs (YAML)
│   ├── goat_molhiv.yaml
│   ├── goat_zinc.yaml
│   ├── exphormer_molhiv.yaml
│   └── exphormer_zinc.yaml
├── experiments/               # Training scripts
│   ├── train_goat.py
│   └── train_exphormer.py
├── notebooks/                 # Development notebooks
└── ROADMAP.md                # Detailed work plan
```

## 🚀 Quick Start

### 1. Test Models on CPU (Small Dataset)

```python
import torch
from juliusz.models.goat import GOAT
from juliusz.utils.positional_encodings import add_positional_encodings
from torch_geometric.data import Data

# Create small test graph
x = torch.randn(20, 32)  # 20 nodes, 32 features
edge_index = torch.randint(0, 20, (2, 40))  # 40 edges
data = Data(x=x, edge_index=edge_index)

# Add positional encodings
data = add_positional_encodings(data, pe_type='laplacian', pe_dim=8)

# Create GOAT model
model = GOAT(
    in_channels=32,
    hidden_channels=64,  # Small for CPU testing
    out_channels=2,
    num_layers=2,
    pe_dim=8,
)

# Forward pass
out = model(data)
print(f"Output shape: {out.shape}")  # [1, 2]
```

### 2. Train GOAT on MolHIV

```bash
# From the juliusz directory
cd /Users/jwasieleski/Prywatne/jul/workspace/machine-learning-project/juliusz

# Train on CPU (for testing)
python experiments/train_goat.py --config configs/goat_molhiv.yaml --device cpu

# Or if you have GPU access:
python experiments/train_goat.py --config configs/goat_molhiv.yaml --device cuda
```

### 3. Train Exphormer on ZINC

```bash
python experiments/train_exphormer.py --config configs/exphormer_zinc.yaml --device cpu
```

## ⚙️ Configuration

Edit YAML files in `configs/` to change hyperparameters:

```yaml
# configs/goat_molhiv.yaml
model:
  model_type: goat
  hidden_channels: 256
  num_layers: 4
  num_heads: 8
  pe_type: laplacian
  pe_dim: 8

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
  eval_metric: rocauc
  track_complexity: true  # Teacher's requirement!
```

## 📊 Complexity Analysis

The code automatically tracks:
- **Memory usage**: Peak GPU/CPU memory
- **Training time**: Per epoch, per batch
- **Model parameters**: Total and trainable

Results are saved in `logs/<experiment_name>/results.json`

## 🔍 Testing on CPU

For development and debugging on CPU:

```python
# Use smaller models
config.model.hidden_channels = 64  # Instead of 256
config.model.num_layers = 2        # Instead of 4

# Use small dataset
from src.utils.data import load_molhiv_dataset
dataset, split = load_molhiv_dataset()
small_dataset = dataset[:100]  # Only 100 graphs

# Test forward pass
model.eval()
with torch.no_grad():
    for i in range(5):
        out = model(dataset[i])
        print(f"Graph {i}: output shape {out.shape}")
```

## 📝 Next Steps

### Week 1 (Current)
- [x] Set up infrastructure
- [x] Implement GOAT
- [x] Implement Exphormer
- [x] Test forward passes on CPU
- [ ] Read papers in detail
- [ ] Test training loop on small dataset

### Week 2
- [ ] Debug and optimize models
- [ ] Test different positional encodings
- [ ] Coordinate with Person B for GPU testing
- [ ] Start complexity analysis

### Week 3-4
- [ ] Full training runs (on Person B's GPU)
- [ ] Ablation studies
- [ ] Complexity comparison with baselines

## 🤝 Coordination with Person B

### What to Share
1. **Model files**: `models/goat.py`, `models/exphormer.py`
2. **Training scripts**: `experiments/train_*.py`
3. **Configs**: `configs/*.yaml`
4. **Usage instructions**: This README

### What to Request
1. GPU access for testing (coordinate times)
2. Full training runs with multiple seeds
3. Baseline model results for comparison

## 📚 Key Papers

1. **GOAT**: Kong et al. "GOAT: A Global Transformer on Large-scale Graphs" (ICML 2023)
   - [Paper Link](https://proceedings.mlr.press/v202/kong23a.html)

2. **Exphormer**: Shirzad et al. "Exphormer: Sparse Transformers for Graphs" (ICML 2023)
   - [Paper Link](https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf)

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd /Users/jwasieleski/Prywatne/jul/workspace/machine-learning-project

# Run with Python path
PYTHONPATH=. python juliusz/experiments/train_goat.py --config juliusz/configs/goat_molhiv.yaml
```

### Out of Memory on CPU
```python
# Reduce batch size
config.training.batch_size = 8

# Reduce model size
config.model.hidden_channels = 64
config.model.num_layers = 2
```

### Slow Training on CPU
```python
# Use smaller dataset for testing
dataset = dataset[:500]  # Only 500 graphs

# Reduce num_workers
config.training.num_workers = 0  # No multiprocessing
```

## 📧 Contact

Person A: Juliusz
Person B: [Your partner's name]

See `ROADMAP.md` for detailed week-by-week plan!


