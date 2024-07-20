#! /bin/zsh
set -e
pdflatex presentation.tex
biber presentation
pdflatex presentation.tex
