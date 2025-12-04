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
├── pyproject.toml               # Poetry dependencies and project config
├── poetry.lock                  # Poetry lock file (generated)
└── README.md                    # This file
```

## Setup (Cross-Platform)

### Prerequisites

- Python 3.8+ (any installation method: system Python, pyenv, etc.)
- Poetry (install from [poetry.pypa.io](https://python-poetry.org/docs/#installation))

### 1. Install Poetry

```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -
```

Add Poetry to your PATH (see Poetry installation instructions for your OS).

### 2. Install Dependencies

Poetry will automatically create a virtual environment and install all dependencies:

```bash
poetry install
```

> [!NOTE]  
> VS Code will automatically detect the Poetry virtual environment. You can also find it with `poetry env info --path`.

### 3. Run Jupyter Notebook

```bash
# Activate the Poetry shell (optional, Poetry can run commands directly)
poetry shell
jupyter notebook main.ipynb

# Or run directly without activating shell
poetry run jupyter notebook main.ipynb
```

## References

- [5] Kong, D., et al. "GOAT: A Global Transformer on Large-scale Graphs." ICML 2023. [Link](https://proceedings.mlr.press/v202/kong23a.html)
- [6] Shirzad, H., et al. "Exphormer: Sparse Transformers for Graphs." ICML 2023. [PDF](https://proceedings.mlr.press/v202/shirzad23a/shirzad23a.pdf)
- [7] Zhang, Y., et al. "G2LFormer: Global-to-Local Attention Scheme in Graph Transformers." 2025. [PDF](https://arxiv.org/pdf/2509.14863)
