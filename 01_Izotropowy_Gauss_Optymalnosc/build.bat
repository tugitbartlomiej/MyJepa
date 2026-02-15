@echo off
cd /d "%~dp0"

echo === Kompilacja LaTeX (przebieg 1/2) ===
pdflatex -interaction=nonstopmode LeJEPA-Wyjasnienie.tex
if errorlevel 1 (
    echo BLAD: Kompilacja nie powiodla sie!
    pause
    exit /b 1
)

echo === Kompilacja LaTeX (przebieg 2/2 - referencje) ===
pdflatex -interaction=nonstopmode LeJEPA-Wyjasnienie.tex

echo === Sprzatanie plikow tymczasowych ===
del /q *.aux *.log *.out *.toc 2>nul

echo === Gotowe: LeJEPA-Wyjasnienie.pdf ===
pause
