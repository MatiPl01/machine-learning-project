# Article Compilation Instructions

## Quick Start

**Important**: Run the compilation script from the `article/` directory!

### On Linux/Mac:
```bash
cd article
chmod +x compile_paper.sh
./compile_paper.sh
```

### On Windows (Git Bash):
```bash
cd article
bash compile_paper.sh
```

### Manual Compilation:
```bash
cd article
pdflatex article.tex
pdflatex article.tex  # Run twice for references
```

## Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages (usually included):
  - `amsmath`, `amssymb` (math)
  - `graphicx` (figures)
  - `hyperref` (links)
  - `booktabs` (tables)
  - `geometry` (margins)

## Article Style

This article is written in a Medium-style format:
- ✅ Accessible, conversational language
- ✅ Clear explanations with analogies
- ✅ Practical takeaways for practitioners
- ✅ Visual formatting optimized for reading
- ✅ No formal academic structure (no chapters, just sections)

## Figures

The article references two figures from the parent directory:
- `../model_comparison.png` - Model comparison plot
- `../learning_curves.png` - Learning curves

These paths are already configured in `article.tex`. The figures should be in the project root directory.

## Current Status

The article includes:
- ✅ Engaging introduction
- ✅ Clear explanation of the problem
- ✅ Comprehensive results and analysis
- ✅ Practical takeaways
- ✅ Conclusion with key insights
- ✅ References

## Notes

- The article is ~7 pages
- All results are from `results_zinc_gpu.json`
- Written in an accessible, Medium-style format
- Perfect for sharing on blogs or Medium
