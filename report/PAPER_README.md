# Paper Compilation Instructions

## Quick Start

**Important**: Run the compilation script from the `report/` directory!

### On Linux/Mac:
```bash
cd report
chmod +x compile_paper.sh
./compile_paper.sh
```

### On Windows (Git Bash):
```bash
cd report
bash compile_paper.sh
```

### Manual Compilation:
```bash
cd report
pdflatex paper.tex
pdflatex paper.tex  # Run twice for references
```

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages (usually included):
  - `amsmath`, `amssymb` (math)
  - `graphicx` (figures)
  - `hyperref` (links)
  - `booktabs` (tables)
  - `geometry` (margins)

## Adding Figures

The paper references two figures from the parent directory:
- `../model_comparison.png` - Model comparison plot
- `../learning_curves.png` - Learning curves

These paths are already configured in `paper.tex`. The figures should be in the project root directory.

## Current Status

The paper includes:
- ✅ Abstract
- ✅ Introduction
- ✅ Related Work
- ✅ Methods (all models described)
- ✅ Experimental Setup
- ✅ Results (complete table with all models)
- ✅ Analysis (complexity-accuracy tradeoffs)
- ✅ Conclusion
- ✅ References

## Next Steps

1. **Add figures**: Uncomment and add figure references in the Results section
2. **Add more analysis**: Include homophily/heterophily analysis if you have data
3. **Add ablations**: Include positional encoding ablation studies
4. **Multiple seeds**: Report mean ± std over multiple runs
5. **More datasets**: Add results on ogbg-molhiv or other datasets

## Figure Placement

To add figures, uncomment these sections in `paper.tex`:

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{model_comparison.png}
\caption{Model comparison showing validation MAE, training time, and parameter counts.}
\label{fig:complexity}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{learning_curves.png}
\caption{Learning curves showing training loss and validation MAE over epochs for all models.}
\label{fig:learning_curves}
\end{figure}
```

## Notes

- The paper is currently ~8-10 pages (depending on content)
- All results are from `results_zinc_gpu.json`
- The paper follows the project requirements from `ML_projects_2025.pdf`
- You may want to add more detailed analysis sections based on your specific findings
