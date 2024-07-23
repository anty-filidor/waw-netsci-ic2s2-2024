.PHONY: main FORCE

main: presentation.pdf

presentation.pdf: FORCE
	pdflatex  -synctex=1 -interaction=nonstopmode -file-line-error -recorder presentation.tex
