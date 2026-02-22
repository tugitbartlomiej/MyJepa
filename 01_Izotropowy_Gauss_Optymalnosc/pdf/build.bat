@echo off
cd /d "%~dp0\..\latex"

echo === Kompilacja LaTeX (przebieg 1/2) ===
pdflatex -interaction=nonstopmode LeJEPA-Wyjasnienie.tex
if errorlevel 1 (
    echo BLAD: Kompilacja nie powiodla sie!
    pause
    exit /b 1
)

echo === Kompilacja LaTeX (przebieg 2/2 - referencje) ===
pdflatex -interaction=nonstopmode LeJEPA-Wyjasnienie.tex

echo === Kopiowanie PDF do pdf/ ===
copy /y LeJEPA-Wyjasnienie.pdf "%~dp0LeJEPA-Wyjasnienie.pdf" >nul

echo === Sprzatanie plikow tymczasowych ===
del /q *.aux *.log *.out *.toc *.pdf 2>nul

echo === Gotowe: pdf\LeJEPA-Wyjasnienie.pdf ===
pause
