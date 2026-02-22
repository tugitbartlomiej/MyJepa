"""Generate plots for variance theory and decision boundary explanation."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# Plot 1: Decision boundary — simple 2D classification
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Generate two classes
N = 80
class_a = np.random.randn(N, 2) * 0.7 + np.array([-1.2, 1.0])
class_b = np.random.randn(N, 2) * 0.7 + np.array([1.2, -0.8])

# Left: clean separation
ax = axes[0]
ax.scatter(class_a[:, 0], class_a[:, 1], c='#2196F3', s=30, alpha=0.7, edgecolors='k', linewidths=0.3, label='Klasa A (incision)')
ax.scatter(class_b[:, 0], class_b[:, 1], c='#F44336', s=30, alpha=0.7, edgecolors='k', linewidths=0.3, label='Klasa B (phaco)')
# Decision boundary
xx = np.linspace(-4, 4, 100)
ax.plot(xx, -xx * 0.85 + 0.1, 'k-', linewidth=2.5, label='Granica decyzyjna')
ax.fill_between(xx, -xx * 0.85 + 0.1, 5, alpha=0.05, color='blue')
ax.fill_between(xx, -xx * 0.85 + 0.1, -5, alpha=0.05, color='red')
ax.text(-2.5, 2.5, 'Strona A', fontsize=12, color='#1565C0', fontweight='bold')
ax.text(1.5, -2.5, 'Strona B', fontsize=12, color='#B71C1C', fontweight='bold')
# New point
ax.scatter([0.5], [1.0], c='#4CAF50', s=150, marker='*', zorder=5, edgecolors='k', linewidths=1)
ax.annotate('Nowy punkt\n= Klasa A!', xy=(0.5, 1.0), xytext=(2.0, 2.5),
            fontsize=10, arrowprops=dict(arrowstyle='->', lw=1.5, color='green'),
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_xlim(-4, 4)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_title('Granica decyzyjna = linia\ndzieląca klasy', fontsize=12, fontweight='bold')
ax.set_xlabel(r'$z_1$ (embedding dim 1)', fontsize=11)
ax.set_ylabel(r'$z_2$ (embedding dim 2)', fontsize=11)
ax.legend(fontsize=8, loc='lower left')

# Middle: isotropic — boundary works in any direction
ax = axes[1]
angles = [0, 45, 90, 135]
colors_line = ['#1565C0', '#4CAF50', '#E65100', '#7B1FA2']
samples_iso = np.random.randn(200, 2)
ax.scatter(samples_iso[:, 0], samples_iso[:, 1], c='#90CAF9', s=15, alpha=0.5)
for ang, col in zip(angles, colors_line):
    rad = np.radians(ang)
    dx, dy = np.cos(rad), np.sin(rad)
    ax.plot([-3*dx, 3*dx], [-3*dy, 3*dy], color=col, linewidth=2, alpha=0.7)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_title(r'Izotropowy: granica działa' + '\n' + r'w KAŻDYM kierunku', fontsize=12, fontweight='bold')
ax.set_xlabel(r'$z_1$', fontsize=11)
ax.set_ylabel(r'$z_2$', fontsize=11)
ax.text(0, -3.0, r'Równo punktów po obu stronach $\forall$ kierunek',
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Right: anisotropic — some directions have no separation
ax = axes[2]
samples_aniso = np.random.randn(200, 2) @ np.array([[2.5, 0], [0, 0.3]])
ax.scatter(samples_aniso[:, 0], samples_aniso[:, 1], c='#FFCC80', s=15, alpha=0.5)
# Good direction
ax.plot([0, 0], [-3.5, 3.5], color='#4CAF50', linewidth=2.5, label=r'Dobry kierunek ($z_1$)')
# Bad direction
ax.plot([-3.5, 3.5], [0, 0], color='#F44336', linewidth=2.5, linestyle='--', label=r'Zły kierunek ($z_2$)')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_title(r'Anizotropowy: granica w $z_2$' + '\n' + 'prawie nie separuje!', fontsize=12, fontweight='bold')
ax.set_xlabel(r'$z_1$', fontsize=11)
ax.set_ylabel(r'$z_2$', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.text(0, -3.0, r'Wzdłuż $z_2$ punkty na kupce $\Rightarrow$ brak separacji',
        fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='#FFCDD2'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'decision_boundary.pdf'), bbox_inches='tight', dpi=300)
print("Plot 1 (decision boundary) saved.")

# ============================================================
# Plot 2: Variance — what is it visually
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: Low variance
ax = axes[0]
data_low = np.random.randn(50) * 0.5 + 5
ax.scatter(data_low, np.zeros_like(data_low), c='#2196F3', s=40, alpha=0.7, zorder=3)
ax.axvline(5, color='red', linewidth=2, linestyle='--', label=r'$\mu = 5$')
for d in data_low[:15]:
    ax.plot([d, 5], [0, 0], color='gray', linewidth=0.5, alpha=0.4)
ax.set_xlim(2, 8)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_title(r'Mała wariancja: $\sigma^2 = 0.25$' + '\nPunkty blisko średniej', fontsize=12, fontweight='bold')
ax.set_xlabel('Wartość', fontsize=11)
ax.legend(fontsize=10)

# Middle: High variance
ax = axes[1]
data_high = np.random.randn(50) * 2.0 + 5
ax.scatter(data_high, np.zeros_like(data_high), c='#FF9800', s=40, alpha=0.7, zorder=3)
ax.axvline(5, color='red', linewidth=2, linestyle='--', label=r'$\mu = 5$')
for d in data_high[:15]:
    ax.plot([d, 5], [0, 0], color='gray', linewidth=0.5, alpha=0.4)
ax.set_xlim(2, 8)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_title(r'Duża wariancja: $\sigma^2 = 4.0$' + '\nPunkty daleko od średniej', fontsize=12, fontweight='bold')
ax.set_xlabel('Wartość', fontsize=11)
ax.legend(fontsize=10)

# Right: Formula visualization
ax = axes[2]
data_ex = np.array([2, 3, 5, 7, 8])
mu_ex = data_ex.mean()
ax.scatter(data_ex, np.zeros_like(data_ex), c='#4CAF50', s=80, zorder=3, edgecolors='k')
ax.axvline(mu_ex, color='red', linewidth=2, linestyle='--')
for i, d in enumerate(data_ex):
    diff = d - mu_ex
    ax.annotate('', xy=(d, 0.15), xytext=(mu_ex, 0.15),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    ax.text((d + mu_ex)/2, 0.22, rf'${diff:+.0f}$', fontsize=10, ha='center', color='purple')
    ax.text(d, -0.15, rf'$x_{i+1}={d}$', fontsize=10, ha='center')
ax.text(mu_ex, -0.35, rf'$\mu = {mu_ex}$', fontsize=11, ha='center', color='red', fontweight='bold')

var_val = np.var(data_ex)
ax.text(5, 0.4, rf'$\sigma^2 = \frac{{1}}{{N}}\sum(x_i - \mu)^2 = {var_val:.1f}$',
        fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 0.6)
ax.set_yticks([])
ax.set_title('Wariancja = średni kwadrat\nodległości od średniej', fontsize=12, fontweight='bold')
ax.set_xlabel('Wartość', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'variance_explanation.pdf'), bbox_inches='tight', dpi=300)
print("Plot 2 (variance) saved.")

# ============================================================
# Plot 3: Covariance — positive, negative, zero
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

configs = [
    (r'Kowariancja dodatnia: $\sigma_{12} > 0$' + '\n' + r'$z_1$ rośnie $\Rightarrow$ $z_2$ rośnie',
     np.array([[1, 0.8], [0.8, 1]]), '#4CAF50'),
    (r'Kowariancja zerowa: $\sigma_{12} = 0$' + '\n' + r'$z_1$ i $z_2$ niezależne',
     np.array([[1, 0], [0, 1]]), '#2196F3'),
    (r'Kowariancja ujemna: $\sigma_{12} < 0$' + '\n' + r'$z_1$ rośnie $\Rightarrow$ $z_2$ maleje',
     np.array([[1, -0.8], [-0.8, 1]]), '#F44336'),
]

for ax, (title, cov, color) in zip(axes, configs):
    samples = np.random.multivariate_normal([0, 0], cov, 200)
    ax.scatter(samples[:, 0], samples[:, 1], c=color, s=20, alpha=0.5, edgecolors='k', linewidths=0.2)

    # Eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    for i in range(2):
        ev = eigvecs[:, i] * np.sqrt(eigvals[i]) * 2
        ax.annotate('', xy=ev, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$z_1$', fontsize=11)
    ax.set_ylabel(r'$z_2$', fontsize=11)

    cov_val = cov[0, 1]
    ax.text(0, -3.0, rf'$\mathrm{{Cov}}(z_1, z_2) = {cov_val:.1f}$',
            fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'covariance_types.pdf'), bbox_inches='tight', dpi=300)
print("Plot 3 (covariance) saved.")

# ============================================================
# Plot 4: Covariance matrix — eigenvalues = shape
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cov_configs = [
    (r'$\Sigma = I$: $\lambda_1 = \lambda_2 = 1$',
     np.array([[1, 0], [0, 1]]), 'Blues'),
    (r'$\Sigma = \mathrm{diag}(4, 0.25)$: $\lambda_1=4, \lambda_2=0.25$',
     np.array([[4, 0], [0, 0.25]]), 'Oranges'),
    (r'$\Sigma$ obrócona: $\lambda_1=3, \lambda_2=0.5$',
     np.array([[1.75, 1.25], [1.25, 1.75]]), 'Greens'),
]

for ax, (title, cov, cmap) in zip(axes, cov_configs):
    samples = np.random.multivariate_normal([0, 0], cov, 300)
    ax.scatter(samples[:, 0], samples[:, 1], c=cmap[:-1].lower(), s=12, alpha=0.4)

    eigvals, eigvecs = np.linalg.eigh(cov)

    # Draw ellipse axes with eigenvalue labels
    for i in range(2):
        ev = eigvecs[:, i] * np.sqrt(eigvals[i]) * 2
        ax.annotate('', xy=ev, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax.annotate('', xy=-ev, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3, alpha=0.3))
        # Label
        offset = np.array([0.15, 0.15])
        ax.text(ev[0] + offset[0], ev[1] + offset[1],
                rf'$\lambda_{i+1}={eigvals[i]:.2f}$',
                fontsize=11, color='red', fontweight='bold')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$z_1$', fontsize=11)
    ax.set_ylabel(r'$z_2$', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'eigenvalues_shape.pdf'), bbox_inches='tight', dpi=300)
print("Plot 4 (eigenvalues) saved.")

print("\nAll variance/decision plots generated!")
