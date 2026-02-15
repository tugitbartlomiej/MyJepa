# TODO: Wyprowadzenie wzoru Gaussa od zera

## Cel
Zastąpić obecną tabelkę "Skąd ten wzór? Rozbijamy go na kawałki" (sekcja 5, linie ~2815–2854) pełnym matematycznym wyprowadzeniem, które pokaże **skąd bierze się każdy element** wzoru:

$$p(z) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)$$

a nie tylko opisuje co robi.

## Lokalizacja w dokumencie
- **Plik**: `isotropic_gaussian_optimal.tex`
- **Sekcja 5**: "Wyprowadzenie: rozkład Gaussa od 1D do 3D" (`\label{sec:wyprowadzenie}`)
- **Podsekja**: "Punkt wyjścia: Gauss w 1D" -- tu dodać wyprowadzenie PRZED tabelką

## Kroki wyprowadzenia

### 1. Punkt wyjścia: jakie właściwości chcemy od rozkładu?
- Aksjomat 1: $p(z) \geq 0$ i $\int p(z)\,dz = 1$ (prawdopodobieństwo)
- Aksjomat 2: rozkład zależy tylko od odległości od średniej $|z - \mu|$ (symetria)
- Aksjomat 3: rozkład ma zadaną średnią $\mu$ i wariancję $\sigma^2$
- Aksjomat 4: maksymalizuje entropię (najmniej założeń o danych)

### 2. Wyprowadzenie kształtu $e^{-\alpha(z-\mu)^2}$ z maksymalizacji entropii
- Zdefiniować entropię $H[p] = -\int p(z) \ln p(z)\,dz$
- Sformułować problem optymalizacyjny z ograniczeniami (mnożniki Lagrange'a)
- Wyprowadzić, że rozwiązanie ma postać $p(z) \propto \exp(-\lambda(z-\mu)^2)$
- Wyjaśnić skąd bierze się kwadrat (a nie np. $|z-\mu|$ czy $(z-\mu)^4$)

### 3. Wyznaczenie stałej $\alpha = \frac{1}{2\sigma^2}$ -- skąd dwójka
- Wymóg: $\text{Var}[Z] = \int (z-\mu)^2 p(z)\,dz = \sigma^2$
- Obliczenie całki Gaussowskiej $\int z^2 e^{-\alpha z^2}dz$ za pomocą triku z pochodną
- Pokazanie, że $\alpha = \frac{1}{2\sigma^2}$ to jedyne rozwiązanie dające wariancję $= \sigma^2$

### 4. Wyznaczenie stałej normalizacyjnej $\frac{1}{\sqrt{2\pi\sigma^2}}$
- Wymóg: $\int_{-\infty}^{\infty} p(z)\,dz = 1$
- Całka Gaussowska: $\int e^{-\alpha z^2}dz = \sqrt{\pi/\alpha}$ (trik z biegunowym)
- Podstawienie $\alpha = \frac{1}{2\sigma^2}$ → stała $= \frac{1}{\sqrt{2\pi\sigma^2}}$
- Wyjaśnienie skąd $\pi$ (z przejścia na współrzędne biegunowe)

### 5. Złożenie wszystkiego -- końcowy wzór
- Połączenie kroków 2–4 w finalny wzór
- Zachowanie obecnej tabelki "kawałki" jako podsumowania (skrócona)
- Zachowanie obecnych figur (`gauss_decomposition.pdf`, `gauss_1d.pdf`)
- Zachowanie reguły 1-2-3 sigma i przykładów liczbowych

### 6. Opcjonalnie: wykres/diagram
- Wizualizacja: jak zmiana $\alpha$ wpływa na kształt (seria krzywych)
- Schemat: "aksjomat → mnożnik Lagrange'a → kształt → normalizacja → wzór"

## Czego NIE ruszać
- Dalszej części sekcji 5 (wyprowadzenie 2D, 3D, K-wymiarowe) -- jest OK
- Sekcji 4 "Teoria wariancji" -- jest OK, dostarcza potrzebne pojęcia
- Istniejących figur -- zachować
