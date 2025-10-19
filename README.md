# Graph Transformers at Scale: Global, Sparse and Hybrid Attention

This project implements and analyzes scalable graph transformers focusing on three attention mechanisms:

1. **GOAT**: Global attention with approximate global patterns
2. **Exphormer**: Sparse attention using expander graph sparsity
3. **G2LFormer**: Hybrid global-to-local attention

**Research Goal**: Compare complexity vs. accuracy tradeoffs across different attention mechanisms on graph benchmarks (OGB, ZINC, Peptides).

## Project Structure

```
├── src/                         # Python modules
│   └── utils/                   # Utility functions
│       └── data.py              # Data loading and analysis
├── main.ipynb                   # Main notebook with analysis
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Setup (Cross-Platform)

### 1. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

> [!NOTE]  
> VS Code will automatically detect the correct Python interpreter in the virtual environment.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

## References

- [5] Kong, D., et al. "GOAT: A Global Transformer on Large-scale Graphs." ICML 2023. [Link](https://proceedings.mlr.press/v202/kong23a.html)
- [6] Shirzad, H., et al. "Exphormer: Sparse Transformers for Graphs." ICML 2023. [PDF](https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf)
- [7] Zhang, Y., et al. "G2LFormer: Global-to-Local Attention Scheme in Graph Transformers." 2025. [PDF](https://arxiv.org/pdf/2509.14863)
