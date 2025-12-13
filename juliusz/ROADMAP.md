# Person A (Juliusz) - Work Roadmap

## 🎯 Your Responsibilities

1. **Implement GOAT & Exphormer Models**
2. **Build Training Infrastructure**
3. **Complexity Analysis Tools**
4. **Positional Encodings**

---

## 📅 Timeline (6 Weeks)

### Week 1: Foundations & GOAT
- [ ] Set up positional encoding utilities (Laplacian, Random Walk)
- [ ] Implement GOAT model architecture
- [ ] Test GOAT forward pass on small data (CPU)
- [ ] Read Exphormer paper & expander graph theory

### Week 2: Exphormer & Training Infrastructure
- [ ] Implement Exphormer model with sparse attention
- [ ] Build training loop with proper optimization
- [ ] Add validation/testing pipelines
- [ ] Implement checkpointing system
- [ ] Test both models end-to-end on CPU (small dataset)

### Week 3: Complexity Analysis
- [ ] Memory profiling utilities
- [ ] Training time measurement
- [ ] FLOPs counting (if feasible)
- [ ] Create complexity comparison framework
- [ ] Coordinate with Person B for GPU testing

### Week 4: Refinement & Ablations
- [ ] Test different positional encoding schemes
- [ ] Ablate attention mechanisms
- [ ] Optimize hyperparameters
- [ ] Bug fixing and code cleanup

### Week 5: Final Experiments
- [ ] Hand over models to Person B for full GPU training
- [ ] Support Person B with any model issues
- [ ] Run complexity analysis on final models
- [ ] Create complexity plots and tables

### Week 6: Report Writing
- [ ] Write Introduction section
- [ ] Write Related Work section
- [ ] Write Methods section (GOAT & Exphormer)
- [ ] Create architecture diagrams
- [ ] Write Complexity Analysis results

---

## 📁 Directory Structure

```
juliusz/
├── ROADMAP.md                      # This file
├── models/                         # Model implementations
│   ├── __init__.py
│   ├── goat.py                     # GOAT implementation
│   ├── exphormer.py                # Exphormer implementation
│   ├── base.py                     # Base transformer class
│   └── layers.py                   # Common layers (attention, etc.)
├── training/                       # Training infrastructure
│   ├── __init__.py
│   ├── trainer.py                  # Main training loop
│   ├── config.py                   # Configuration dataclasses
│   └── utils.py                    # Training utilities
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── positional_encodings.py     # PE implementations
│   ├── complexity.py               # Complexity analysis
│   └── metrics.py                  # Evaluation metrics
├── configs/                        # Experiment configs
│   ├── goat_molhiv.yaml
│   ├── goat_zinc.yaml
│   ├── exphormer_molhiv.yaml
│   └── exphormer_zinc.yaml
├── experiments/                    # Experiment scripts
│   ├── train_goat.py
│   ├── train_exphormer.py
│   └── analyze_complexity.py
└── notebooks/                      # Development notebooks
    ├── test_goat.ipynb
    └── test_exphormer.ipynb
```

---

## 🔑 Key Implementation Details

### GOAT (Global Transformer)
- **Approximate global attention** using virtual nodes
- **O(N) complexity** instead of O(N²)
- Key components:
  - Global attention pooling
  - Virtual super nodes
  - Message passing between local and global

### Exphormer (Sparse Transformer)
- **Expander graphs** for sparse attention
- Uses virtual expander graph overlay
- Key components:
  - Expander graph construction (d-regular)
  - Sparse attention via expander edges
  - Local + expander attention combination

### Positional Encodings
1. **Laplacian Eigenvectors** (spectral)
2. **Random Walk** (structural)
3. **Degree centrality** (simple)
4. **Learnable embeddings** (baseline)

### Training Infrastructure
- Multi-GPU support (even though you'll test on CPU)
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping
- Checkpointing (save best model)

---

## 🔬 Complexity Metrics to Track

1. **Memory Usage**
   - Peak GPU memory (Person B will measure)
   - Model parameters count
   - Activation memory

2. **Time Complexity**
   - Training time per epoch
   - Inference time per batch
   - Forward pass breakdown

3. **Computational Complexity**
   - FLOPs (floating point operations)
   - Attention complexity (theoretical)

---

## 📚 Papers to Read

### Essential (Read First)
1. **GOAT Paper**: "GOAT: A Global Transformer on Large-scale Graphs" (ICML 2023)
   - Focus on: Virtual node mechanism, complexity analysis
   
2. **Exphormer Paper**: "Exphormer: Sparse Transformers for Graphs" (ICML 2023)
   - Focus on: Expander graph theory, sparse attention construction

### Background (Skim)
3. **Graph Transformers**: "A Generalization of Transformer Networks to Graphs" (AAAI 2021)
4. **Positional Encodings**: "Rethinking Graph Transformers with Spectral Attention" (NeurIPS 2021)

### Theory (For Understanding)
5. **Expander Graphs**: Wikipedia article on expander graphs
6. **Graph Laplacian**: Spectral graph theory basics

---

## 🤝 Coordination with Person B

### What You'll Provide to Person B:
1. Working model implementations (`.py` files)
2. Training scripts that work on CPU
3. Configuration files for experiments
4. Documentation on how to use your code

### What You'll Need from Person B:
1. GPU access for testing (coordinate times)
2. Baseline model results for comparison
3. Full training runs with multiple seeds
4. Final benchmark results for complexity analysis

### Weekly Check-ins:
- **Monday**: Share progress, blockers, plans
- **Thursday**: Code review, integration testing
- **Weekend**: Coordinate GPU access if needed

---

## 💻 Development Tips (CPU-focused)

1. **Use Small Datasets for Testing**
   ```python
   debug_dataset = dataset[:50]  # Only 50 graphs
   ```

2. **Smaller Models for Debugging**
   ```python
   hidden_dim = 32  # Instead of 256
   num_layers = 2   # Instead of 8
   ```

3. **Profile Your Code**
   ```python
   import cProfile
   cProfile.run('train_model()')
   ```

4. **Use Assertions**
   ```python
   assert attention_weights.sum(dim=-1).allclose(torch.ones(...))
   ```

5. **Visualize Attention**
   - Plot attention matrices (small graphs)
   - Verify sparse patterns in Exphormer
   - Check global attention in GOAT

---

## 🚀 Getting Started (Today!)

1. ✅ Read this roadmap
2. ⬜ Skim GOAT and Exphormer papers (2 hours)
3. ⬜ Implement positional encodings (2 hours)
4. ⬜ Start GOAT base architecture (2-3 hours)
5. ⬜ Test forward pass on 1 small graph (1 hour)

**Goal for Week 1**: GOAT model working on CPU with small dataset

---

## 📝 Notes

- Focus on **correct implementation** over optimization
- Write **clean, documented code** (Person B will use it)
- Test **incrementally** (don't write everything at once)
- Ask for help early if stuck
- Keep complexity analysis in mind while implementing

Good luck! 🚀

