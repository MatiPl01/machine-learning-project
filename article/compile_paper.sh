#!/bin/bash
# Script to compile the LaTeX article
# Run this from the article/ directory

echo "Compiling LaTeX article from article/ directory..."

# Compile with pdflatex (run twice for references)
pdflatex -interaction=nonstopmode article.tex
pdflatex -interaction=nonstopmode article.tex

# Clean up auxiliary files (but keep PDF)
rm -f *.aux *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls *.bbl *.blg *.bcf *.run.xml

echo ""
echo "Done! Article compiled to article/article.pdf"
echo "Temporary LaTeX files have been cleaned up."
