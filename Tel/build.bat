@echo off
echo Building mobile-friendly LeJEPA PDF...
pdflatex -interaction=nonstopmode LeJEPA-Telefon.tex
pdflatex -interaction=nonstopmode LeJEPA-Telefon.tex
echo.
echo Done! Output: LeJEPA-Telefon.pdf
pause
