# Graph Neural Networks vs Graph Transformers: Experimental Comparison

This repository implements and compares:
- **classical graph neural networks (GNNs)**, 
- **hybrid models** (GNN + Virtual Node), 
- **graph transformers** 

on molecular and graph-level prediction tasks from the Open Graph Benchmark (OGB) and ZINC.

---

## 1. Project Aim

**Question:** Does global attention (as in Graph Transformers) improve prediction over local message-passing (GNNs) on graph-level tasks?

We compare:

- **Baseline GNNs** — GCN, GAT, GIN, GraphMLP (local or no structure).
- **Hybrid models** — GCN and GIN with a Virtual Node for global information exchange.
- **Graph Transformers** — GOAT (global attention via virtual nodes) and Exphormer (sparse attention via expander graphs).

All models are trained and evaluated on the **same datasets and hyperparameter regime**:

| Parametr | Tryb CPU | Tryb GPU |
|----------|----------|----------|
| **hidden_dim** | 64 | 256 |
| **num_layers** | 3 | 5 |
| **num_heads** | 4 | 8 |
| **dropout** | 0.1 | 0.1 |
| **learning_rate** | 1e-3 | 1e-4 |
| **batch_size** | 32 | 64 |
| **num_epochs** | 10 | 200 |
| **pe_dim** (positional enc.) | 8 | 16 |
| **early_stopping patience** | 5 | 20 |

 so that differences in performance reflect architecture rather than tuning.

**References:**

- Kong et al., *GOAT: A Global Transformer on Large-scale Graphs* (ICML 2023) — `papers/kong23a (1).pdf`
- Shirzad et al., *Exphormer: Sparse Transformers for Graphs* (ICML 2023) — `papers/2303.06147v2.pdf`

---

## 2. Implemented Models

| Model | Type | Complexity | Description |
|-------|------|------------|-------------|
| **GCN** | Baseline | O(E) | Graph Convolutional Network (Kipf & Welling, ICLR 2017) |
| **GAT** | Baseline | O(E) | Graph Attention Network — attention over neighbors |
| **GIN** | Baseline | O(E) | Graph Isomorphism Network — sum aggregation, strong expressivity |
| **GraphMLP** | Baseline | O(N) | MLP on node features only (no graph structure) |
| **GCN+VN** | Hybrid | O(E+N) | GCN + Virtual Node for global context |
| **GIN+VN** | Hybrid | O(E+N) | GIN + Virtual Node + Jumping Knowledge |
| **GOAT** | Transformer | O(N) | Global attention via virtual super-nodes |
| **Exphormer** | Transformer | O(Nd) | Sparse attention via expander graphs |

Code: `models/baselines.py`, `models/hybrid.py`, `models/goat.py`, `models/exphormer.py`.

---

## 3. Datasets

| Dataset | Task | Size | Metric |
|---------|------|------|--------|
| **ZINC** | Regression (molecular property) | ~12K graphs | MAE (lower is better) |
| **ogbg-molhiv** | Binary classification (HIV inhibition) | ~41K graphs | ROC-AUC (higher is better) |
| **ogbg-molpcba** | Multi-label classification (128 tasks) | ~438K graphs | Average Precision (AP, higher is better) |

Data loading: `src/utils/data.py`.

---

## 4. Experimental Setup

- **Hardware:** GPU (e.g. NVIDIA RTX 3090) for full runs.
- **Shared hyperparameters (GPU mode):** `hidden_dim=256`, `num_layers=5`, `num_heads=8`, `dropout=0.1`, `pe_dim=16` (for transformers), early stopping (patience 20), model-specific learning rates (e.g. higher for GIN, GOAT, Exphormer).
- **Reproducibility:** Experiments are run in Jupyter notebooks; results and checkpoints are saved under `experiments/` and `checkpoints/`.

Notebooks:

- `experiments/compare_all_models zinc.ipynb`
- `experiments/compare_all_models molhiv.ipynb`
- `experiments/compare_all_models molpcba.ipynb`

---

## 5. Results Summary

### 5.1 ZINC (Regression — MAE ↓)

| Rank | Model | Val MAE | Params |
|------|-------|--------|--------|
| 1 | **GIN+VN** | **0.355** | ~1.52M |
| 2 | GIN | 0.368 | ~0.93M |
| 3 | GCN+VN | 0.475 | ~0.93M |
| 4 | GOAT | 0.530 | ~5.01M |
| 5 | Exphormer | 0.651 | ~5.34M |
| 6 | GAT | 0.693 | ~0.33M |
| 7 | GCN | 0.829 | ~0.33M |
| 8 | GraphMLP | 0.831 | ~0.26M |

![img.png](img.png)
![img_1.png](img_1.png)

**Takeaway:** GIN and GIN+VN clearly outperform the rest. Virtual Node helps (GCN+VN vs GCN). Transformers (GOAT, Exphormer) are worse than the best GNNs despite more parameters.

---

### 5.2 ogbg-molhiv (Binary classification — ROC-AUC ↑)

| Rank | Model | Val ROC-AUC | Params |
|------|-------|-------------|--------|
| 1 | **GIN** | **0.811** | ~0.93M |
| 2 | GCN+VN | 0.804 | ~0.93M |
| 3 | GAT | 0.801 | ~0.34M |
| 4 | GCN | 0.791 | ~0.33M |
| 5 | GIN+VN | 0.791 | ~1.52M |
| 6 | GOAT | 0.676 | ~5.01M |
| 7 | Exphormer | 0.662 | ~5.34M |
| 8 | GraphMLP | 0.636 | ~0.27M |

![img_2.png](img_2.png)
![img_3.png](img_3.png)

**Takeaway:** Best results are from GIN and GCN+VN. GAT and GCN are close. Transformers (GOAT, Exphormer) lag behind the top GNNs. Virtual Node helps GCN (GCN+VN > GCN) but not clearly GIN on this dataset.

---

### 5.3 ogbg-molpcba (Multi-label — AP ↑)

| Rank | Model | Val AP | Params |
|------|-------|--------|--------|
| 1 | **GIN** | **0.484** | ~0.96M |
| 2 | GCN | 0.478 | ~0.37M |
| 3 | GAT | 0.429 | ~0.37M |
| 4 | Exphormer | 0.427 | ~5.37M |
| 5 | GOAT | 0.418 | ~5.04M |
| 6 | GCN+VN | 0.340 | ~0.96M |
| 7 | GIN+VN | 0.328 | ~1.56M |
| 8 | GraphMLP | 0.251 | ~0.30M |

![img_4.png](img_4.png)
![img_5.png](img_5.png)

**Takeaway:** GIN and GCN are best; GAT and both transformers are mid-pack. On this large multi-label dataset, Virtual Node variants (GCN+VN, GIN+VN) perform worse than their non-VN counterparts, suggesting possible overfitting or need for different hyperparameters.

---

## 6. Conclusions

1. **GIN is the strongest single architecture** across ZINC, molhiv, and molpcba — consistent with its higher expressive power (WL-1 style).
2. **Virtual Node helps GCN** (ZINC, molhiv) by adding global context at low cost; effect on GIN is dataset-dependent (helps on ZINC, mixed on molhiv, hurts on molpcba in current setup).
3. **Graph Transformers (GOAT, Exphormer) do not beat the best GNNs** here — they use more parameters and compute but achieve worse or similar validation metrics. Possible reasons: molecule graphs are small and local structure may be enough; our training/hyperparameters may favor GNNs; or the tasks do not require strong long-range reasoning.
4. **GraphMLP is consistently worst**, showing that **using graph structure is essential**.
5. **Efficiency:** Best validation scores often come from GIN or GCN+VN with ~1M parameters, while transformers use ~5M and train longer — so the best accuracy/compute trade-off in these experiments is with GNNs and hybrids.

These results support the claim that **for these molecular/graph-level benchmarks, carefully chosen GNNs (and simple hybrids) can match or outperform graph transformers**, and that global attention is not always necessary for good performance.
