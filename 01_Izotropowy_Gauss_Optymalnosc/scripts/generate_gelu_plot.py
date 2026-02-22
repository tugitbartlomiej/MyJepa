"""Generate GELU activation function plot comparing with ReLU."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
PNG_DIR = os.path.join(SCRIPT_DIR, '..', 'png_preview')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

x = np.linspace(-4, 4, 1000)

# GELU: x * Phi(x) where Phi is the standard normal CDF
gelu = x * norm.cdf(x)

# ReLU for comparison
relu = np.maximum(0, x)

# Identity for reference
identity = x

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: GELU vs ReLU ---
ax = axes[0]
ax.plot(x, gelu, 'b-', linewidth=2.5, label='GELU(x) = x · Φ(x)')
ax.plot(x, relu, 'r--', linewidth=2, alpha=0.7, label='ReLU(x) = max(0, x)')
ax.plot(x, identity, 'gray', linewidth=1, alpha=0.4, linestyle=':', label='y = x (liniowa)')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

ax.set_xlabel('x (wartość wejściowa)', fontsize=12)
ax.set_ylabel('f(x) (wartość wyjściowa)', fontsize=12)
ax.set_title('GELU vs ReLU: funkcje aktywacji', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)
ax.set_ylim(-1.5, 4)

# Annotate key regions
ax.annotate('ReLU: ostro zeruje\nwszystko < 0',
            xy=(-2, 0), xytext=(-3.5, 1.5),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='red'),
            color='red')

ax.annotate('GELU: łagodnie\ntłumi (nie zeruje!)',
            xy=(-1.5, gelu[np.argmin(np.abs(x - (-1.5)))]),
            xytext=(-3, -1),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue'),
            color='blue')

# --- Right: What GELU does to a distribution ---
ax2 = axes[1]

# Show input distribution (normal-ish)
np.random.seed(42)
inputs = np.random.randn(5000) * 1.5

# Apply GELU
gelu_outputs = inputs * norm.cdf(inputs)

ax2.hist(inputs, bins=60, alpha=0.4, color='gray', label='Wejście (przed GELU)')
ax2.hist(gelu_outputs, bins=60, alpha=0.6, color='blue', label='Wyjście (po GELU)')
ax2.axvline(0, color='black', linewidth=0.5)

ax2.set_xlabel('Wartość neuronu', fontsize=12)
ax2.set_ylabel('Liczba wartości (z 5000)', fontsize=12)
ax2.set_title('Efekt GELU na rozkład 5000 wartości', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Annotations
ax2.annotate('Wartości ujemne\nzostają stłumione\n(ściągnięte w stronę 0)',
            xy=(-0.3, 200), xytext=(-3, 600),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue'))

ax2.annotate('Duży pik przy 0:\nGELU "zepchnęła"\nujemne wartości tutaj',
            xy=(0.05, 800), xytext=(2.5, 700),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            color='blue', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'gelu_activation.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(PNG_DIR, 'gelu_activation.png'), dpi=150, bbox_inches='tight')
print("GELU plot saved.")
