#!/bin/bash
# Script to compile the LaTeX paper
# Run this from the paper/ directory

echo "Compiling LaTeX paper from paper/ directory..."

# Compile with pdflatex (run twice for references)
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex

# Clean up auxiliary files (but keep PDF)
rm -f *.aux *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls *.bbl *.blg *.bcf *.run.xml

echo ""
echo "Done! Paper compiled to paper/paper.pdf"
echo "Temporary LaTeX files have been cleaned up."
