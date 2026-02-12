"""Generate example vs counter-example: isotropic vs anisotropic embeddings
for downstream classification tasks."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap

np.random.seed(42)

# ============================================================
# Plot: 2 rows x 2 cols
# Row 1: Isotropic embeddings — Task A (vertical boundary) and Task B (horizontal boundary)
# Row 2: Anisotropic embeddings — same two tasks
# Show: both tasks work with isotropic, only one works with anisotropic
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 13))

N = 150  # points per class

# --- Generate isotropic embeddings (two classes) ---
# Class 0: centered at (-1.2, 0)
# Class 1: centered at (1.2, 0)
iso_class0 = np.random.randn(N, 2) * 1.0 + np.array([-1.2, 0])
iso_class1 = np.random.randn(N, 2) * 1.0 + np.array([1.2, 0])

# --- Generate anisotropic embeddings (squished along z2) ---
# Same centers, but sigma_1 = 3.5, sigma_2 = 0.12
aniso_class0 = np.random.randn(N, 2) * np.array([3.5, 0.12]) + np.array([-1.2, 0])
aniso_class1 = np.random.randn(N, 2) * np.array([3.5, 0.12]) + np.array([1.2, 0])

# --- Task A: boundary is VERTICAL (separating along z1) ---
# --- Task B: boundary is HORIZONTAL (separating along z2) ---
# For Task B, re-center classes vertically
iso_class0_B = np.random.randn(N, 2) * 1.0 + np.array([0, -1.2])
iso_class1_B = np.random.randn(N, 2) * 1.0 + np.array([0, 1.2])

aniso_class0_B = np.random.randn(N, 2) * np.array([3.5, 0.12]) + np.array([0, -0.3])
aniso_class1_B = np.random.randn(N, 2) * np.array([3.5, 0.12]) + np.array([0, 0.3])


def compute_accuracy(c0, c1, axis):
    """Simple linear probe accuracy along given axis (0=z1, 1=z2)."""
    # Find optimal threshold
    all_points = np.vstack([c0, c1])
    all_labels = np.array([0]*len(c0) + [1]*len(c1))

    vals = all_points[:, axis]
    threshold = (c0[:, axis].mean() + c1[:, axis].mean()) / 2

    if c0[:, axis].mean() < c1[:, axis].mean():
        preds = (vals > threshold).astype(int)
    else:
        preds = (vals < threshold).astype(int)

    return (preds == all_labels).mean() * 100


def plot_panel(ax, c0, c1, boundary_dir, title, accuracy, is_good):
    """Plot one panel with embeddings and decision boundary."""
    ax.scatter(c0[:, 0], c0[:, 1], c='#2196F3', alpha=0.5, s=25, label='Klasa A (incision)')
    ax.scatter(c1[:, 0], c1[:, 1], c='#F44336', alpha=0.5, s=25, label='Klasa B (phaco)')

    xlim = ax.get_xlim() if ax.get_xlim() != (0, 1) else (-5, 5)
    ylim = ax.get_ylim() if ax.get_ylim() != (0, 1) else (-5, 5)

    # Set consistent limits
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4, 4)

    # Draw decision boundary
    if boundary_dir == 'vertical':
        threshold = (c0[:, 0].mean() + c1[:, 0].mean()) / 2
        ax.axvline(threshold, color='green', linewidth=2.5, linestyle='--',
                   label=f'Granica decyzyjna')
        # Shade regions
        ax.axvspan(-6, threshold, alpha=0.05, color='#2196F3')
        ax.axvspan(threshold, 6, alpha=0.05, color='#F44336')
    else:
        threshold = (c0[:, 1].mean() + c1[:, 1].mean()) / 2
        ax.axhline(threshold, color='green', linewidth=2.5, linestyle='--',
                   label=f'Granica decyzyjna')
        ax.axvspan(-6, 6, ymin=0, ymax=0.5, alpha=0.05, color='#2196F3')

    # Accuracy badge
    badge_color = '#4CAF50' if is_good else '#F44336'
    badge_text = f'Accuracy: {accuracy:.0f}%'
    ax.text(0.98, 0.98, badge_text, transform=ax.transAxes, fontsize=14,
            fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=badge_color,
                      alpha=0.9, edgecolor='white'),
            color='white')

    ax.set_xlabel(r'$z_1$', fontsize=13)
    ax.set_ylabel(r'$z_2$', fontsize=13)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


# --- Row 1: ISOTROPIC ---
acc_iso_A = compute_accuracy(iso_class0, iso_class1, axis=0)
acc_iso_B = compute_accuracy(iso_class0_B, iso_class1_B, axis=1)

plot_panel(axes[0, 0], iso_class0, iso_class1, 'vertical',
           'IZOTROPOWY + Zadanie A\n(granica pionowa wzdłuż $z_1$)',
           acc_iso_A, True)

plot_panel(axes[0, 1], iso_class0_B, iso_class1_B, 'horizontal',
           'IZOTROPOWY + Zadanie B\n(granica pozioma wzdłuż $z_2$)',
           acc_iso_B, True)

# --- Row 2: ANISOTROPIC ---
acc_aniso_A = compute_accuracy(aniso_class0, aniso_class1, axis=0)
acc_aniso_B = compute_accuracy(aniso_class0_B, aniso_class1_B, axis=1)

plot_panel(axes[1, 0], aniso_class0, aniso_class1, 'vertical',
           'ANIZOTROPOWY + Zadanie A\n(granica pionowa wzdłuż $z_1$)',
           acc_aniso_A, acc_aniso_A > 80)

plot_panel(axes[1, 1], aniso_class0_B, aniso_class1_B, 'horizontal',
           'ANIZOTROPOWY + Zadanie B\n(granica pozioma wzdłuż $z_2$)',
           acc_aniso_B, acc_aniso_B > 80)

# Row labels
fig.text(0.02, 0.73, 'IZOTROPOWY\n' + r'$\Sigma = \mathbf{I}$',
         fontsize=14, fontweight='bold', color='#4CAF50',
         ha='center', va='center', rotation=90)
fig.text(0.02, 0.28, 'ANIZOTROPOWY\n' + r'$\sigma_1 \gg \sigma_2$',
         fontsize=14, fontweight='bold', color='#F44336',
         ha='center', va='center', rotation=90)

# Big arrows between rows
fig.text(0.5, 0.50, r'$\Downarrow$ Co się zmienia, gdy rozkład jest anizotropowy?',
         fontsize=13, ha='center', va='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout(rect=[0.05, 0, 1, 1])
plt.savefig('figures/isotropy_example_vs_counterexample.pdf', bbox_inches='tight')
plt.savefig('png_preview/isotropy_example_vs_counterexample.png', dpi=150, bbox_inches='tight')
print("Plot saved.")
