"""Generate GELU activation function plot comparing with ReLU."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

ax2.hist(inputs, bins=60, alpha=0.4, color='gray', label='Wejście (przed GELU)', density=True)
ax2.hist(gelu_outputs, bins=60, alpha=0.6, color='blue', label='Wyjście (po GELU)', density=True)
ax2.axvline(0, color='black', linewidth=0.5)

ax2.set_xlabel('Wartość', fontsize=12)
ax2.set_ylabel('Gęstość', fontsize=12)
ax2.set_title('Efekt GELU na rozkład wartości', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Annotations
ax2.annotate('Wartości ujemne\nzostają stłumione\n(nie wycięte!)',
            xy=(-0.5, 0.1), xytext=(-3, 0.5),
            fontsize=9, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig('figures/gelu_activation.pdf', bbox_inches='tight')
plt.savefig('png_preview/gelu_activation.png', dpi=150, bbox_inches='tight')
print("GELU plot saved.")
