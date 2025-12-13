# Comprehensive Experiment Plan for Graph Transformers Research

## 📊 Current Notebook Status
Your `benchmark_comparison.ipynb` is a **good starting point** but needs expansion for publication-quality research.

**What it has** ✓:
- Basic GOAT vs Exphormer comparison
- Training curves (loss, MAE)
- Complexity metrics (memory, time)
- Simple tradeoff plot

**What's missing for a research paper**:
- Multiple datasets
- Statistical significance (multiple seeds)
- Ablation studies
- Baseline comparisons
- Scalability analysis
- Homophily/heterophily analysis

---

## 🔬 Experiment Design: What Experienced Researchers Measure

### **Project Requirements Coverage**

From your project description:
1. ✓ **Complexity vs. accuracy tradeoffs** - partially done
2. ✗ **Homophily/heterophily robustness** - NOT DONE YET
3. ✗ **Ablations on positional encodings** - NOT DONE YET
4. ✗ **Ablations on global-token designs** - NOT DONE YET

From teacher's notes:
1. ✓ ROC-AUC / Accuracy
2. ✓ Memory tracking
3. ✓ Training time
4. ✗ Comparison with baselines (GCN, GAT)

---

## 📋 Comprehensive Experiment Suite

### **Tier 1: Core Experiments** (Must Have)
These are essential for your report.

#### **Experiment 1: Multi-Dataset Comparison**
**What**: Test GOAT vs Exphormer on all 3 datasets
**Datasets**: 
- ZINC (regression, ~23 nodes avg)
- MolHIV (binary classification, ~18 nodes avg)
- Peptides-func (regression, ~173 nodes avg, LONG RANGE!)

**Why**: Shows generalization across:
- Task types (classification vs regression)
- Graph sizes (small vs large)
- Domains (molecules vs peptides)

**Metrics to track**:
- ROC-AUC (MolHIV)
- MAE (ZINC, Peptides)
- Training time per epoch
- Peak memory usage
- Convergence speed (epochs to best val)

**Expected runtime**:
- CPU: 2-3 hours per dataset
- GPU: 20-40 minutes per dataset

---

#### **Experiment 2: Statistical Significance**
**What**: Multiple seeds for confidence intervals
**Setup**: Run each model 3-5 times with different seeds

**Why**: 
- Publications require error bars / confidence intervals
- Shows results aren't due to lucky initialization
- Enables statistical tests (t-test, ANOVA)

**Report format**:
```
GOAT on ZINC:  0.387 ± 0.012 MAE
Exphormer:     0.391 ± 0.015 MAE
```

**Expected runtime**:
- CPU: 5-8 hours total
- GPU: 1-2 hours total

---

#### **Experiment 3: Baseline Comparison**
**What**: Compare against standard GNNs
**Baselines**: GCN, GAT (Person B should implement these)

**Why**: Shows benefit of transformer architectures

**Comparison table**:
```
Model          | ZINC MAE | MolHIV AUC | Params | Time/epoch
---------------|----------|------------|--------|------------
GCN            |   ?      |     ?      |   ?    |    ?
GAT            |   ?      |     ?      |   ?    |    ?
GOAT (Ours)    |   ?      |     ?      |   ?    |    ?
Exphormer      |   ?      |     ?      |   ?    |    ?
```

**Expected runtime**:
- Coordinate with Person B
- They run baselines on GPU: 1-2 hours

---

### **Tier 2: Ablation Studies** (Important)
These answer "why does it work?"

#### **Experiment 4: Positional Encoding Ablation**
**What**: Test different PE types
**Variants**:
1. No PE (baseline)
2. Laplacian eigenvectors (spectral)
3. Random walk (structural)
4. Degree centrality (simple)
5. Learned embeddings (trainable)

**Why**: Shows which PE works best and why

**Expected results plot**:
```
         ZINC MAE
Laplacian:  0.387
Random Walk: 0.392
Degree:      0.401
No PE:       0.415
```

**Expected runtime**:
- CPU: 4-5 hours (5 variants)
- GPU: 40-60 minutes

---

#### **Experiment 5: Model Scale Ablation**
**What**: Test different model sizes
**Variants**:
- Small: 64 hidden dim, 2 layers
- Medium: 128 hidden dim, 4 layers
- Large: 256 hidden dim, 8 layers

**Why**: Shows scalability and overfitting behavior

**Plot**: Accuracy vs Parameters (Pareto frontier)

**Expected runtime**:
- CPU: Not recommended (too slow for large)
- GPU: 1-2 hours

---

#### **Experiment 6: Exphormer Degree Ablation**
**What**: Test different expander degrees
**Variants**: d = 2, 4, 8, 16

**Why**: Shows sparsity vs accuracy tradeoff

**Expected plot**:
```
Degree 2:  Fast but lower accuracy
Degree 4:  Good balance (recommended)
Degree 8:  Slower, marginal improvement
Degree 16: Approaching full attention
```

**Expected runtime**:
- CPU: 2-3 hours
- GPU: 30-40 minutes

---

### **Tier 3: Advanced Analysis** (Publication Quality)
These make your report stand out.

#### **Experiment 7: Scalability Analysis**
**What**: How do models scale with graph size?
**Setup**: 
- Create subsets of different sizes
- Measure time and memory vs number of nodes

**Plot**: 
- X-axis: Number of nodes
- Y-axis: Time per batch (log scale)
- Lines: GOAT (should be linear), Exphormer (should be linear), Full Attention (quadratic)

**Why**: Validates theoretical complexity claims

**Expected runtime**:
- CPU: 1-2 hours
- GPU: 20-30 minutes

---

#### **Experiment 8: Homophily/Heterophily Analysis**
**What**: How do models perform on different graph structures?
**Setup**:
1. Compute homophily ratio for each dataset
2. Correlate with model performance
3. Analyze which model works better for which structure

**Homophily ratio**:
```python
# Ratio of edges connecting same-label nodes
h = (edges_same_label) / (total_edges)
```

**Expected findings**:
- High homophily → Local attention might suffice
- Low homophily → Global attention helps more

**Expected runtime**:
- Analysis only: 30 minutes
- Combined with main experiments

---

#### **Experiment 9: Attention Visualization**
**What**: Visualize learned attention patterns
**Setup**:
1. Extract attention weights from models
2. Visualize on sample graphs
3. Compare GOAT vs Exphormer attention

**Why**: Provides interpretability and insights

**Plots**:
- Attention matrices (heatmaps)
- Graph with attention edges highlighted
- Attention distribution statistics

**Expected runtime**:
- 1-2 hours (mostly coding)

---

#### **Experiment 10: Convergence Analysis**
**What**: How fast do models converge?
**Metrics**:
- Epochs to 95% of best performance
- Training stability (variance across epochs)
- Learning curve shape

**Plot**: Validation metric vs wall-clock time (not epochs!)

**Why**: Shows practical efficiency

**Expected runtime**:
- Combined with main experiments

---

## 🖥️ Two-Configuration Strategy

### **Configuration A: CPU Quick Test** ⚡
**Purpose**: Verify code works, plots render correctly
**Time**: 15-20 minutes

```python
# CPU Configuration
CPU_CONFIG = {
    'dataset_size': 500,           # Small subset
    'batch_size': 16,              # Small batches
    'num_epochs': 10,              # Quick test
    'hidden_dim': 32,              # Tiny model
    'num_layers': 2,
    'num_heads': 2,
    'num_seeds': 1,                # Single run
    'datasets': ['zinc'],          # One dataset only
}
```

**What to check**:
- ✓ Code runs without errors
- ✓ Plots render correctly
- ✓ Metrics make sense (not NaN, reasonable range)
- ✓ Memory tracking works
- ✓ File saving works

---

### **Configuration B: GPU Full Experiments** 🚀
**Purpose**: Production runs for report
**Time**: 6-8 hours total

```python
# GPU Configuration
GPU_CONFIG = {
    'dataset_size': 'full',        # Full dataset
    'batch_size': 128,             # Large batches
    'num_epochs': 200,             # Full training
    'hidden_dim': 256,             # Full model
    'num_layers': 6,
    'num_heads': 8,
    'num_seeds': 5,                # Statistical significance
    'datasets': ['zinc', 'molhiv', 'peptides'],
}
```

**Execution plan**:
1. Run overnight or over weekend
2. Save all checkpoints
3. Log to wandb or tensorboard
4. Generate all plots automatically

---

## 📊 Enhanced Plots for Research Paper

### **Current plots** (keep these):
1. ✓ Training loss curves
2. ✓ Validation metric curves
3. ✓ Time per epoch
4. ✓ Complexity vs accuracy scatter

### **Add these plots**:

#### **Plot 5: Multi-Dataset Performance**
```
Bar chart with error bars (from multiple seeds)
X-axis: Dataset (ZINC, MolHIV, Peptides)
Y-axis: Performance metric
Bars: GOAT, Exphormer, GCN, GAT
```

#### **Plot 6: Scalability Curves**
```
Log-log plot
X-axis: Number of nodes
Y-axis: Time per batch (ms)
Lines: GOAT (O(N)), Exphormer (O(Nd)), Theory lines
```

#### **Plot 7: Ablation Heatmap**
```
Heatmap showing performance across:
- Rows: PE types
- Columns: Model sizes
- Color: Validation metric
```

#### **Plot 8: Pareto Frontier**
```
Scatter plot
X-axis: Parameters (or FLOPs)
Y-axis: Accuracy
Points: Different configurations
Show Pareto frontier curve
```

#### **Plot 9: Learning Efficiency**
```
X-axis: Wall-clock time (minutes)
Y-axis: Validation metric
Lines: Different models
Shows which reaches good performance fastest
```

#### **Plot 10: Memory vs Accuracy**
```
Scatter plot
X-axis: Peak memory (MB)
Y-axis: Best validation metric
Points: Models and configurations
```

#### **Plot 11: Homophily Analysis**
```
Scatter plot
X-axis: Graph homophily ratio
Y-axis: Model performance
Points: Individual graphs
Color: Model type
Regression line for each model
```

#### **Plot 12: Attention Pattern Visualization**
```
Grid of attention heatmaps
Rows: GOAT, Exphormer
Columns: Different sample graphs
Show which nodes attend to which
```

---

## 📝 Results Table Template

### **Table 1: Main Results**
```
Model         | ZINC MAE ↓    | MolHIV AUC ↑  | Peptides MAE ↓ | Params | Time/epoch
--------------|---------------|---------------|----------------|--------|------------
GCN           | 0.420 ± 0.015 | 0.752 ± 0.008 | 0.287 ± 0.012  | 500K   | 8.2s
GAT           | 0.405 ± 0.012 | 0.768 ± 0.006 | 0.275 ± 0.010  | 650K   | 12.5s
GOAT (ours)   | 0.387 ± 0.010 | 0.781 ± 0.005 | 0.254 ± 0.008  | 720K   | 15.3s
Exphormer     | 0.391 ± 0.011 | 0.779 ± 0.006 | 0.258 ± 0.009  | 680K   | 13.8s
```
↓ = lower is better, ↑ = higher is better

### **Table 2: Complexity Comparison**
```
Model      | Attention | Time      | Memory    | FLOPs
-----------|-----------|-----------|-----------|----------
Full Attn  | O(N²)     | 45.2s     | 8.2 GB    | 12.5 GF
GOAT       | O(N)      | 15.3s     | 2.1 GB    | 4.2 GF
Exphormer  | O(Nd)     | 13.8s     | 1.9 GB    | 3.8 GF
```

### **Table 3: Ablation Study - Positional Encodings**
```
PE Type       | ZINC MAE | MolHIV AUC | Improvement
--------------|----------|------------|-------------
No PE         | 0.415    | 0.745      | baseline
Degree        | 0.401    | 0.758      | +2.5%
Random Walk   | 0.392    | 0.771      | +5.8%
Laplacian     | 0.387    | 0.781      | +7.2% ✓
```

### **Table 4: Expander Degree Ablation**
```
Degree | ZINC MAE | Time/epoch | Memory | Tradeoff
-------|----------|------------|--------|----------
d=2    | 0.398    | 11.2s      | 1.5 GB | Fast, less accurate
d=4    | 0.391    | 13.8s      | 1.9 GB | Balanced ✓
d=8    | 0.389    | 18.4s      | 2.4 GB | Slow, marginal gain
d=16   | 0.388    | 26.1s      | 3.2 GB | Diminishing returns
```

---

## 🎯 Recommended Execution Plan

### **Phase 1: CPU Validation** (Today - 30 min)
1. Run `benchmark_comparison.ipynb` with CPU config
2. Verify all plots render
3. Check metrics are reasonable
4. Fix any bugs

### **Phase 2: GPU Core Experiments** (Tomorrow - 4 hours)
1. Multi-dataset comparison (ZINC, MolHIV, Peptides)
2. Multiple seeds (3-5 runs each)
3. Generate main results table

### **Phase 3: Ablation Studies** (Day 3-4 - 3 hours)
1. PE ablation (5 types)
2. Model scale ablation (3 sizes)
3. Expander degree ablation (4 values)

### **Phase 4: Advanced Analysis** (Day 5-6 - 2 hours)
1. Scalability curves
2. Homophily analysis
3. Attention visualization

### **Phase 5: Baselines** (Coordinate with Person B)
1. Get GCN/GAT results
2. Add to comparison tables
3. Discuss results together

---

## 💾 File Organization

```
juliusz/
├── notebooks/
│   ├── benchmark_comparison.ipynb           # Your current notebook (good!)
│   ├── full_experiments.ipynb              # New: GPU experiments
│   ├── ablation_studies.ipynb              # New: Ablations
│   ├── scalability_analysis.ipynb          # New: Scalability
│   └── visualization.ipynb                 # New: Attention viz
├── results/
│   ├── cpu_test/                           # Quick CPU runs
│   ├── main_results/                       # Core GPU experiments
│   ├── ablations/                          # Ablation studies
│   └── figures/                            # All plots for paper
└── configs/
    ├── cpu_quick_test.yaml
    └── gpu_full_experiments.yaml
```

---

## 🔍 What Makes This Research-Grade?

1. **Statistical rigor**: Multiple seeds, error bars, significance tests
2. **Comprehensive comparison**: Multiple datasets, baselines, ablations
3. **Theoretical validation**: Complexity analysis matches theory
4. **Practical insights**: Homophily analysis, scalability curves
5. **Reproducibility**: Configs, seeds, detailed setup
6. **Visualization quality**: Publication-ready plots
7. **Ablation thoroughness**: Every design choice validated

---

## 📊 Your Current Notebook: Verdict

**Status**: Good foundation, needs expansion

**Keep**:
- ✓ Basic structure
- ✓ Training loop with complexity tracking
- ✓ Simple comparison plots

**Add**:
- More datasets (MolHIV, Peptides)
- Multiple seeds
- More plots (see above)
- Ablation studies
- Statistical tests

**Recommendation**:
1. Keep current notebook as "quick test"
2. Create new "full_experiments.ipynb" for GPU
3. Create separate notebooks for ablations

---

## 🎓 Expected Outcomes for Report

After running all experiments, you'll have:

1. **Main results**: GOAT vs Exphormer on 3 datasets ✓
2. **Baseline comparison**: vs GCN, GAT ✓
3. **Ablations**: PE types, model sizes, expander degrees ✓
4. **Complexity analysis**: Time, memory, scalability ✓
5. **Homophily study**: When does each model excel? ✓
6. **12+ publication-quality plots** ✓
7. **4+ results tables with error bars** ✓

This is publication-quality work! 🎉

---

**Next step**: Should I create the GPU configuration notebook and ablation study notebooks for you?

