# Quick Start: Running Your Experiments

## 🎯 Your Current Notebook is Now Ready!

I've updated `benchmark_comparison.ipynb` with a **simple config toggle**.

### How to Use

#### **Option 1: CPU Mode** (Quick Test - 20 minutes)
```python
# In cell 3, set:
EXPERIMENT_MODE = 'cpu'  # ← Quick test
```
Then run all cells. This will:
- Use 500 graphs only
- Small model (64 dim, 3 layers)
- Train for 20 epochs
- Complete in ~20 minutes on CPU
- **Purpose**: Verify everything works

#### **Option 2: GPU Mode** (Full Experiments - 2-4 hours)
```python
# In cell 3, set:
EXPERIMENT_MODE = 'gpu'  # ← Full experiment
```
Then run all cells on GPU. This will:
- Use full ZINC dataset (10,000 graphs)
- Large model (256 dim, 6 layers)
- Train for 200 epochs
- Complete in ~2-4 hours on GPU
- **Purpose**: Get publication-quality results

---

## 📋 Execution Checklist

### **Today (CPU)**
- [ ] Open `benchmark_comparison.ipynb`
- [ ] Set `EXPERIMENT_MODE = 'cpu'`
- [ ] Run all cells
- [ ] Check that plots render correctly
- [ ] Verify metrics are reasonable
- [ ] Save results to show Person B

**Expected output**: 
- 4 training plots
- Comparison table
- `benchmark_comparison.png` saved

### **Tomorrow (GPU - Person B's computer)**
- [ ] Set `EXPERIMENT_MODE = 'gpu'`
- [ ] Run overnight or during class
- [ ] Save all results
- [ ] Compare CPU vs GPU numbers
- [ ] Start ablation studies

---

## 🔬 Next Steps: Advanced Experiments

Once basic benchmark works, expand to:

### **1. Multi-Dataset Experiments** (4 hours GPU)
Add MolHIV and Peptides-func:
```python
datasets = ['zinc', 'molhiv', 'peptides']
for dataset_name in datasets:
    # Run experiments
```

### **2. Ablation Studies** (2 hours GPU)
Test different positional encodings:
```python
for pe_type in ['laplacian', 'random_walk', 'degree', 'none']:
    # Train and compare
```

### **3. Statistical Significance** (3 hours GPU)
Multiple seeds for error bars:
```python
for seed in [42, 123, 456, 789, 1011]:
    torch.manual_seed(seed)
    # Train and record results
```

---

## 📊 What Your Current Notebook Produces

### **Metrics Tracked** ✓
- Training loss & MAE
- Validation loss & MAE
- Time per epoch
- Peak memory usage
- Total parameters

### **Plots Generated** ✓
1. Training loss curves (GOAT vs Exphormer)
2. Validation MAE curves
3. Training time per epoch
4. Complexity vs Accuracy scatter

### **Comparison Table** ✓
```
Metric                    | GOAT      | Exphormer | Winner
Parameters                | X         | Y         | ?
Peak Memory (MB)          | X         | Y         | ?
Total Training Time (s)   | X         | Y         | ?
Best Val MAE              | X         | Y         | ?
```

---

## 🎓 For Your Report

### **From Current Notebook You Get**:

**Section 4: Experiments**
```markdown
### 4.1 Experimental Setup
- Datasets: ZINC (regression)
- Models: GOAT, Exphormer
- Hardware: [CPU/GPU specs]
- Implementation: PyTorch Geometric

### 4.2 Results
See Figure X: Training curves show...
See Table X: GOAT achieves...

### 4.3 Complexity Analysis
Peak memory: GOAT uses X MB vs Exphormer Y MB
Training time: GOAT takes X seconds per epoch...
```

**Figures for Paper**:
- Figure 1: Training curves (from your notebook)
- Figure 2: Complexity vs Accuracy (from your notebook)
- Figure 3: Time comparison (from your notebook)
- Table 1: Main results (from your notebook)
- Table 2: Complexity metrics (from your notebook)

### **What's Still Missing** (for next notebooks):
- Multiple datasets
- Baseline comparisons (wait for Person B)
- Statistical significance (multiple seeds)
- Ablation studies
- Homophily analysis

---

## 💡 Pro Tips

### **Debugging on CPU**
If something fails:
1. Reduce dataset size: `SMALL_SIZE = 100`
2. Reduce epochs: `NUM_EPOCHS = 5`
3. Reduce model: `HIDDEN_DIM = 32`
4. Check one batch: `for data in train_loader: break`

### **Optimizing for GPU**
Once on GPU:
1. Increase batch size: `BATCH_SIZE = 256`
2. Use mixed precision: `torch.cuda.amp`
3. Pin memory: `pin_memory=True` in DataLoader
4. Multiple GPUs: `torch.nn.DataParallel`

### **Saving Results**
Add this after experiments:
```python
import json

results = {
    'goat': {
        'best_val_mae': best_goat_mae,
        'complexity': goat_complexity.to_dict(),
        'history': goat_history,
    },
    'exphormer': {
        'best_val_mae': best_exphormer_mae,
        'complexity': exphormer_complexity.to_dict(),
        'history': exphormer_history,
    }
}

with open('results_cpu.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 🎯 Success Criteria

### **CPU Run (Today)**
✓ Completes without errors
✓ Plots look reasonable
✓ MAE is in range 0.3-0.5 for ZINC
✓ Time is ~30-60 seconds per epoch
✓ Memory is <2GB

### **GPU Run (Tomorrow)**
✓ Completes full 200 epochs
✓ Better results than CPU (lower MAE)
✓ Faster per epoch (~10-20s)
✓ Converges smoothly
✓ Results are reproducible

---

## 📞 Coordination with Person B

### **Share with Person B**:
1. Your CPU test results (to show progress)
2. The notebook file (so they can run on GPU)
3. Any bugs you found and fixed
4. Estimated GPU runtime

### **Request from Person B**:
1. GCN baseline results
2. GAT baseline results
3. GPU access schedule
4. Their dataset preprocessing approach

### **Together**:
1. Compare GOAT/Exphormer vs GCN/GAT
2. Decide which model works best
3. Plan ablation studies
4. Divide report writing

---

## 📝 Immediate Action Items

**Right now** (10 minutes):
1. ✅ Read `EXPERIMENT_PLAN.md` - comprehensive plan
2. ✅ Open `benchmark_comparison.ipynb`
3. ✅ Set `EXPERIMENT_MODE = 'cpu'`
4. ✅ Run all cells

**If it works** (it should!):
- 🎉 You have a working benchmark!
- 📊 You have 4 publication-quality plots
- 📈 You have complexity metrics (teacher's requirement!)
- ⏭️ Ready for GPU experiments

**If it fails**:
- Check error message
- Try with smaller config
- Share error with me
- Debug together

---

**You're 80% done with implementation! Now it's about running experiments and writing the report.** 🚀

