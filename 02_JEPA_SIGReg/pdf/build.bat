@echo off
cd /d "%~dp0\..\latex"

echo === Kompilacja LaTeX (przebieg 1/2) ===
pdflatex -interaction=nonstopmode LeJEPA-JEPA-SIGReg.tex
if errorlevel 1 (
    echo BLAD: Kompilacja nie powiodla sie!
    pause
    exit /b 1
)

echo === Kompilacja LaTeX (przebieg 2/2 - referencje) ===
pdflatex -interaction=nonstopmode LeJEPA-JEPA-SIGReg.tex

echo === Kopiowanie PDF do pdf/ ===
copy /y LeJEPA-JEPA-SIGReg.pdf "%~dp0LeJEPA-JEPA-SIGReg.pdf" >nul

echo === Sprzatanie plikow tymczasowych ===
del /q *.aux *.log *.out *.toc *.pdf 2>nul

echo === Gotowe: pdf\LeJEPA-JEPA-SIGReg.pdf ===
pause
