"""Generate MLP visualization plots for the LaTeX document."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
PNG_DIR = os.path.join(SCRIPT_DIR, '..', 'png_preview')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

# --- Style ---
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})


def gelu(x):
    return x * norm.cdf(x)


# ================================================================
# FIGURE 1: MLP krok po kroku -- co dzieje sie z wektorem
# ================================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

np.random.seed(42)

# Input vector (d=4 for illustration)
h = np.array([1.2, -0.8, 0.5, -1.5])
d_in = len(h)
d_hidden = 8  # 2x expansion for illustration (real: 4x)

# Random but structured W1 (d_hidden x d_in)
W1 = np.array([
    [ 0.6, -0.3,  0.8,  0.1],
    [-0.2,  0.7,  0.1, -0.5],
    [ 0.4,  0.4, -0.6,  0.3],
    [-0.1, -0.8,  0.2,  0.9],
    [ 0.9,  0.1, -0.3, -0.2],
    [-0.5,  0.6,  0.7,  0.4],
    [ 0.3, -0.4, -0.1,  0.8],
    [ 0.7,  0.2,  0.5, -0.6],
])
b1 = np.array([0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0, -0.1])

# W2 (d_in x d_hidden)
W2 = np.array([
    [ 0.4, -0.2,  0.6,  0.1, -0.3,  0.5,  0.2, -0.4],
    [-0.3,  0.7, -0.1,  0.4,  0.2, -0.6,  0.3,  0.1],
    [ 0.5,  0.1,  0.3, -0.5,  0.7,  0.2, -0.4,  0.3],
    [-0.1,  0.4, -0.2,  0.6, -0.1,  0.3,  0.5, -0.2],
])
b2 = np.array([0.05, -0.05, 0.1, 0.0])

# Compute MLP steps
z1 = W1 @ h + b1           # Step 1: linear expansion
z2 = gelu(z1)               # Step 2: GELU
z3 = W2 @ z2 + b2           # Step 3: linear compression

colors_in = ['#2196F3' if v >= 0 else '#F44336' for v in h]
colors_z1 = ['#2196F3' if v >= 0 else '#F44336' for v in z1]
colors_z2 = ['#2196F3' if v >= 0 else '#F44336' for v in z2]
colors_out = ['#2196F3' if v >= 0 else '#F44336' for v in z3]

# Panel 1: Input h
ax = axes[0]
bars = ax.barh(range(d_in), h, color=colors_in, edgecolor='black', linewidth=0.8)
ax.set_yticks(range(d_in))
ax.set_yticklabels([f'$h_{{{i+1}}}$' for i in range(d_in)], fontsize=12)
ax.set_xlabel('Wartość')
ax.set_title(f'Wejście h\n(d = {d_in})')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
for i, v in enumerate(h):
    ax.text(v + (0.1 if v >= 0 else -0.1), i, f'{v:.1f}',
            ha='left' if v >= 0 else 'right', va='center', fontsize=9, fontweight='bold')

# Panel 2: After W1*h + b1
ax = axes[1]
bars = ax.barh(range(d_hidden), z1, color=colors_z1, edgecolor='black', linewidth=0.8)
ax.set_yticks(range(d_hidden))
ax.set_yticklabels([f'$z_{{{i+1}}}$' for i in range(d_hidden)], fontsize=10)
ax.set_xlabel('Wartość')
ax.set_title(f'Po $W_1 h + b_1$\n(d = {d_hidden}, rozszerzenie)')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
for i, v in enumerate(z1):
    ax.text(v + (0.08 if v >= 0 else -0.08), i, f'{v:.2f}',
            ha='left' if v >= 0 else 'right', va='center', fontsize=8)

# Panel 3: After GELU
ax = axes[2]
bars = ax.barh(range(d_hidden), z2, color=colors_z2, edgecolor='black', linewidth=0.8)
ax.set_yticks(range(d_hidden))
ax.set_yticklabels([f'$\\tilde{{z}}_{{{i+1}}}$' for i in range(d_hidden)], fontsize=10)
ax.set_xlabel('Wartość')
ax.set_title(f'Po GELU\n(nieliniowość)')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
for i, v in enumerate(z2):
    killed = abs(z1[i]) > 0.3 and abs(v) < 0.1
    ax.text(v + (0.08 if v >= 0 else -0.08), i, f'{v:.2f}',
            ha='left' if v >= 0 else 'right', va='center', fontsize=8,
            color='red' if killed else 'black',
            fontweight='bold' if killed else 'normal')

# Annotate killed neurons
killed_indices = [i for i in range(d_hidden) if z1[i] < -0.5 and abs(z2[i]) < 0.15]
if killed_indices:
    ax.annotate('GELU wycisza\nujemne wartości!',
                xy=(z2[killed_indices[0]], killed_indices[0]),
                xytext=(1.2, killed_indices[0] + 1.5),
                fontsize=9, ha='center', color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

# Panel 4: Output
ax = axes[3]
bars = ax.barh(range(d_in), z3, color=colors_out, edgecolor='black', linewidth=0.8)
ax.set_yticks(range(d_in))
ax.set_yticklabels([f'$o_{{{i+1}}}$' for i in range(d_in)], fontsize=12)
ax.set_xlabel('Wartość')
ax.set_title(f'Wyjście $W_2\\tilde{{z}} + b_2$\n(d = {d_in}, kompresja)')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlim(-2.5, 2.5)
for i, v in enumerate(z3):
    ax.text(v + (0.1 if v >= 0 else -0.1), i, f'{v:.2f}',
            ha='left' if v >= 0 else 'right', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mlp_step_by_step.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(PNG_DIR, 'mlp_step_by_step.png'), dpi=150, bbox_inches='tight')
print("Figure 1 saved: mlp_step_by_step")
plt.close()


# ================================================================
# FIGURE 2: MLP architecture diagram (flow diagram with dimensions)
# ================================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

# Define boxes
boxes = [
    (1.0, 2.0, 2.0, 1.5, '$\\mathbf{h}$\n$\\in \\mathbb{R}^{384}$', '#E3F2FD', '#1565C0'),
    (4.0, 2.0, 2.2, 1.5, '$W_1 \\mathbf{h} + b_1$\n$\\in \\mathbb{R}^{1536}$', '#FFF3E0', '#E65100'),
    (7.2, 2.0, 2.0, 1.5, 'GELU\n$\\in \\mathbb{R}^{1536}$', '#F3E5F5', '#6A1B9A'),
    (10.2, 2.0, 2.5, 1.5, '$W_2 (\\cdot) + b_2$\n$\\in \\mathbb{R}^{384}$', '#E8F5E9', '#2E7D32'),
]

for x, y, w, h_box, text, facecolor, edgecolor in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h_box,
                                    boxstyle='round,pad=0.15',
                                    facecolor=facecolor,
                                    edgecolor=edgecolor,
                                    linewidth=2.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h_box/2, text,
            ha='center', va='center', fontsize=12, fontweight='bold')

# Arrows between boxes
arrow_style = dict(arrowstyle='->', color='#333333', lw=2)
ax.annotate('', xy=(4.0, 2.75), xytext=(3.0, 2.75), arrowprops=arrow_style)
ax.annotate('', xy=(7.2, 2.75), xytext=(6.2, 2.75), arrowprops=arrow_style)
ax.annotate('', xy=(10.2, 2.75), xytext=(9.2, 2.75), arrowprops=arrow_style)

# Dimension labels on arrows
ax.text(3.5, 3.6, 'Rozszerzenie\n$384 \\to 1536$\n($4\\times$)',
        ha='center', va='bottom', fontsize=10, color='#E65100', fontweight='bold')
ax.text(6.7, 3.6, 'Nieliniowość\n$1536 \\to 1536$',
        ha='center', va='bottom', fontsize=10, color='#6A1B9A', fontweight='bold')
ax.text(9.7, 3.6, 'Kompresja\n$1536 \\to 384$\n($4\\times$)',
        ha='center', va='bottom', fontsize=10, color='#2E7D32', fontweight='bold')

# Title and parameter counts
ax.text(7.0, 0.8, 'Parametry: $W_1$ (589 824 wag) + $W_2$ (589 824 wag) + biasy (1920) = 1.18M na blok',
        ha='center', va='center', fontsize=10, style='italic', color='#555555')
ax.text(7.0, 0.3, 'Przy 12 blokach: 12 × 1.18M = 14.2M parametrów (ponad połowa ViT-Small!)',
        ha='center', va='center', fontsize=10, style='italic', color='#555555')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mlp_architecture.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(PNG_DIR, 'mlp_architecture.png'), dpi=150, bbox_inches='tight')
print("Figure 2 saved: mlp_architecture")
plt.close()


# ================================================================
# FIGURE 3: What GELU does inside MLP -- before vs after
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

np.random.seed(123)
# Simulate 1000 hidden neurons after W1*h + b1 (roughly normal)
pre_gelu = np.random.randn(1000) * 1.2
post_gelu = gelu(pre_gelu)

# Left: scatter before vs after GELU
ax = axes[0]
ax.scatter(pre_gelu, post_gelu, s=8, alpha=0.5, c='#1565C0')
x_line = np.linspace(-4, 4, 200)
ax.plot(x_line, gelu(x_line), 'r-', linewidth=2, label='GELU(x)', zorder=5)
ax.plot(x_line, x_line, 'gray', linewidth=1, linestyle=':', alpha=0.5, label='y = x')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Przed GELU: $W_1 h + b_1$', fontsize=12)
ax.set_ylabel('Po GELU', fontsize=12)
ax.set_title('Transformacja GELU wewnątrz MLP', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_xlim(-4, 4)
ax.set_ylim(-1.5, 4)

# Shade the "killed" region
ax.axvspan(-4, -1.5, alpha=0.08, color='red')
ax.text(-2.75, 3, 'Neurony\n"wyciszone"\nprzez GELU', fontsize=9,
        ha='center', color='red', fontweight='bold')

# Shade the "passed" region
ax.axvspan(1.5, 4, alpha=0.08, color='green')
ax.text(2.75, 0.5, 'Neurony\n"przepuszczone"\nbez zmian', fontsize=9,
        ha='center', color='green', fontweight='bold')

# Right: histogram comparison
ax = axes[1]
ax.hist(pre_gelu, bins=50, alpha=0.4, color='#90CAF9', label='Przed GELU', edgecolor='#1565C0')
ax.hist(post_gelu, bins=50, alpha=0.6, color='#1565C0', label='Po GELU', edgecolor='#0D47A1')
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Wartość neuronu ukrytej warstwy', fontsize=12)
ax.set_ylabel('Liczba neuronów (z 1000)', fontsize=12)
ax.set_title('Rozkład wartości: przed i po GELU', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

ax.annotate('GELU "ściąga"\nujemne do ~0\n(selekcja cech!)',
            xy=(0, 180), xytext=(2, 120),
            fontsize=10, ha='center', color='#0D47A1', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.5))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mlp_gelu_effect.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(PNG_DIR, 'mlp_gelu_effect.png'), dpi=150, bbox_inches='tight')
print("Figure 3 saved: mlp_gelu_effect")
plt.close()

print("\nAll MLP plots generated successfully!")
