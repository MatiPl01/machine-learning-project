#!/bin/bash
# Script to compile the LaTeX paper

echo "Compiling LaTeX paper..."

# Compile with pdflatex (run twice for references)
pdflatex paper.tex
pdflatex paper.tex

# Clean up auxiliary files
rm -f *.aux *.log *.out *.toc

echo "Done! Paper compiled to paper.pdf"
echo "Note: If you have figures, make sure they are in the same directory or update paths in paper.tex"
