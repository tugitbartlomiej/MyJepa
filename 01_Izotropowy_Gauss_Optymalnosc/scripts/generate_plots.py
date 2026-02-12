"""Generate plots for isotropic vs anisotropic Gaussian explanation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Plot 1: 2D isotropic vs anisotropic Gaussian
# ============================================================
np.random.seed(42)
N = 500

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Isotropic Gaussian N(0, I) ---
ax = axes[0]
samples_iso = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N)
ax.scatter(samples_iso[:, 0], samples_iso[:, 1], alpha=0.4, s=12, c='#2196F3')
ellipse = Ellipse((0, 0), 2, 2, fill=False, edgecolor='#1565C0', linewidth=2.5, linestyle='--')
ax.add_patch(ellipse)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.set_title(r'Izotropowy: $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$', fontsize=14, fontweight='bold')
ax.set_xlabel(r'$z_1$', fontsize=12)
ax.set_ylabel(r'$z_2$', fontsize=12)
# Eigenvalue arrows
ax.annotate('', xy=(1.0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5))
ax.annotate('', xy=(0, 1.0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5))
ax.text(1.1, -0.3, r'$\lambda_1=1$', fontsize=11, color='#D32F2F')
ax.text(-0.9, 1.1, r'$\lambda_2=1$', fontsize=11, color='#D32F2F')

# --- Anisotropic Gaussian ---
ax = axes[1]
cov_aniso = [[3, 0.5], [0.5, 0.3]]
samples_aniso = np.random.multivariate_normal([0, 0], cov_aniso, N)
ax.scatter(samples_aniso[:, 0], samples_aniso[:, 1], alpha=0.4, s=12, c='#FF9800')
# Compute eigenvectors for ellipse
eigvals, eigvecs = np.linalg.eigh(cov_aniso)
angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
ellipse = Ellipse((0, 0), 2*np.sqrt(eigvals[1]), 2*np.sqrt(eigvals[0]),
                  angle=angle, fill=False, edgecolor='#E65100', linewidth=2.5, linestyle='--')
ax.add_patch(ellipse)
# Eigenvector arrows
for i in range(2):
    ev = eigvecs[:, i] * np.sqrt(eigvals[i])
    ax.annotate('', xy=ev, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5))
ax.text(1.2, 0.5, rf'$\lambda_1={eigvals[1]:.1f}$', fontsize=11, color='#D32F2F')
ax.text(-1.5, -0.8, rf'$\lambda_2={eigvals[0]:.1f}$', fontsize=11, color='#D32F2F')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.set_title(r'Anizotropowy: $\mathrm{Cov}(\mathbf{z}) \neq \mathbf{I}$', fontsize=14, fontweight='bold')
ax.set_xlabel(r'$z_1$', fontsize=12)
ax.set_ylabel(r'$z_2$', fontsize=12)

# --- Collapsed (degenerate) ---
ax = axes[2]
samples_collapse = np.random.multivariate_normal([0, 0], [[0.01, 0], [0, 0.01]], N)
ax.scatter(samples_collapse[:, 0], samples_collapse[:, 1], alpha=0.6, s=12, c='#F44336')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.set_title(r'Kolaps: $\mathbf{z} \approx \mathbf{c}$ (degeneracja)', fontsize=14, fontweight='bold')
ax.set_xlabel(r'$z_1$', fontsize=12)
ax.set_ylabel(r'$z_2$', fontsize=12)
ax.text(0.3, 1.5, 'Brak informacji!\nWszystkie embeddingi\nw jednym punkcie',
        fontsize=10, color='#B71C1C', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2', alpha=0.8))

plt.tight_layout()
plt.savefig('isotropic_vs_anisotropic_2d.pdf', bbox_inches='tight', dpi=300)
plt.savefig('isotropic_vs_anisotropic_2d.png', bbox_inches='tight', dpi=200)
print("Plot 1 saved.")

# ============================================================
# Plot 2: 3D isotropic Gaussian surface + density heatmap
# ============================================================
fig = plt.figure(figsize=(12, 5))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z_iso = (1 / (2 * np.pi)) * np.exp(-0.5 * (X**2 + Y**2))

ax1.plot_surface(X, Y, Z_iso, cmap='viridis', alpha=0.85, edgecolor='none')
ax1.set_xlabel(r'$z_1$', fontsize=11)
ax1.set_ylabel(r'$z_2$', fontsize=11)
ax1.set_zlabel(r'$p(\mathbf{z})$', fontsize=11)
ax1.set_title(r'$\mathcal{N}(\mathbf{0}, \mathbf{I}_2)$ — gestość 3D', fontsize=13, fontweight='bold')
ax1.view_init(elev=25, azim=-60)

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z_iso, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax2, label=r'$p(\mathbf{z})$')
# Draw circles for 1σ, 2σ
for r, label in [(1, r'$1\sigma$'), (2, r'$2\sigma$')]:
    circle = plt.Circle((0, 0), r, fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
    ax2.add_patch(circle)
    ax2.text(r * 0.707 + 0.15, r * 0.707 + 0.15, label, color='white', fontsize=11, fontweight='bold')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.set_xlabel(r'$z_1$', fontsize=11)
ax2.set_ylabel(r'$z_2$', fontsize=11)
ax2.set_title(r'Izolinie gestości — okręgi (izotropia!)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('isotropic_gaussian_3d.pdf', bbox_inches='tight', dpi=300)
plt.savefig('isotropic_gaussian_3d.png', bbox_inches='tight', dpi=200)
print("Plot 2 saved.")

# ============================================================
# Plot 3: Bias/variance illustration — linear probe
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

np.random.seed(7)
N_train = 80

# Isotropic case
ax = axes[0]
Z_iso = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], N_train)
true_boundary = np.array([-2, 2])
# True labels based on linear boundary
y_true = (Z_iso @ np.array([1, -0.5]) > 0).astype(int)
colors = ['#2196F3' if yi == 0 else '#F44336' for yi in y_true]
ax.scatter(Z_iso[:, 0], Z_iso[:, 1], c=colors, alpha=0.6, s=25, edgecolors='k', linewidths=0.3)
# True decision boundary
xx = np.linspace(-3, 3, 100)
ax.plot(xx, 2 * xx, 'k-', linewidth=2, label='Prawdziwa granica')
# Learned boundaries from subsamples
for _ in range(10):
    idx = np.random.choice(N_train, 30, replace=False)
    Z_sub = Z_iso[idx]
    y_sub = y_true[idx]
    from numpy.linalg import lstsq
    # Simple linear classifier
    beta, _, _, _ = lstsq(Z_sub, y_sub - 0.5, rcond=None)
    if abs(beta[1]) > 0.01:
        slope = -beta[0] / beta[1]
        ax.plot(xx, slope * xx, color='#4CAF50', alpha=0.3, linewidth=1)
ax.plot([], [], color='#4CAF50', alpha=0.5, linewidth=1.5, label='Nauczone granice')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_title(r'Izotropowy $\Rightarrow$ maly bias i wariancja', fontsize=13, fontweight='bold')
ax.set_xlabel(r'$z_1$', fontsize=11)
ax.set_ylabel(r'$z_2$', fontsize=11)
ax.legend(fontsize=9, loc='upper left')

# Anisotropic case
ax = axes[1]
Z_aniso = np.random.multivariate_normal([0, 0], [[4, 0], [0, 0.15]], N_train)
y_true_a = (Z_aniso @ np.array([1, -0.5]) > 0).astype(int)
colors_a = ['#2196F3' if yi == 0 else '#F44336' for yi in y_true_a]
ax.scatter(Z_aniso[:, 0], Z_aniso[:, 1], c=colors_a, alpha=0.6, s=25, edgecolors='k', linewidths=0.3)
ax.plot(xx, 2 * xx, 'k-', linewidth=2, label='Prawdziwa granica')
for _ in range(10):
    idx = np.random.choice(N_train, 30, replace=False)
    Z_sub = Z_aniso[idx]
    y_sub = y_true_a[idx]
    beta, _, _, _ = lstsq(Z_sub, y_sub - 0.5, rcond=None)
    if abs(beta[1]) > 0.01:
        slope = -beta[0] / beta[1]
        ax.plot(xx, slope * xx, color='#4CAF50', alpha=0.3, linewidth=1)
ax.plot([], [], color='#4CAF50', alpha=0.5, linewidth=1.5, label='Nauczone granice')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_title(r'Anizotropowy $\Rightarrow$ duzy bias i wariancja', fontsize=13, fontweight='bold')
ax.set_xlabel(r'$z_1$', fontsize=11)
ax.set_ylabel(r'$z_2$', fontsize=11)
ax.legend(fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig('bias_variance_illustration.pdf', bbox_inches='tight', dpi=300)
plt.savefig('bias_variance_illustration.png', bbox_inches='tight', dpi=200)
print("Plot 3 saved.")

# ============================================================
# Plot 4: 1D projections — characteristic function idea
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: 1D projections of isotropic vs anisotropic
ax = axes[0]
t = np.linspace(-4, 4, 200)
pdf_std = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * t**2)
pdf_wide = (1/np.sqrt(2*np.pi*3)) * np.exp(-0.5 * t**2 / 3)
pdf_narrow = (1/np.sqrt(2*np.pi*0.3)) * np.exp(-0.5 * t**2 / 0.3)

ax.plot(t, pdf_std, 'b-', linewidth=2.5, label=r'Izotropowy: $\sigma^2=1$ (cel)')
ax.fill_between(t, pdf_std, alpha=0.15, color='blue')
ax.plot(t, pdf_wide, 'r--', linewidth=2, label=r'Anizo. kierunek 1: $\sigma^2=3$')
ax.plot(t, pdf_narrow, 'g-.', linewidth=2, label=r'Anizo. kierunek 2: $\sigma^2=0.3$')
ax.set_xlabel(r'Rzut $\mathbf{a}^\top \mathbf{z}$', fontsize=12)
ax.set_ylabel(r'Gestość', fontsize=12)
ax.set_title('Rzuty 1D na losowe kierunki', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-4, 4)

# Right: Characteristic functions
ax = axes[1]
t_cf = np.linspace(-5, 5, 200)
phi_target = np.exp(-0.5 * t_cf**2)  # N(0,1) CF
phi_wide = np.exp(-0.5 * 3 * t_cf**2)  # N(0,3) CF
phi_narrow = np.exp(-0.5 * 0.3 * t_cf**2)  # N(0,0.3) CF

ax.plot(t_cf, phi_target, 'b-', linewidth=2.5, label=r'$\varphi_{\mathcal{N}(0,1)}(t) = e^{-t^2/2}$ (cel)')
ax.fill_between(t_cf, phi_target, alpha=0.1, color='blue')
ax.plot(t_cf, phi_wide, 'r--', linewidth=2, label=r'$\varphi_{\sigma^2=3}(t)$ — za szeroki')
ax.plot(t_cf, phi_narrow, 'g-.', linewidth=2, label=r'$\varphi_{\sigma^2=0.3}(t)$ — za waski')

# Shade the error
ax.fill_between(t_cf, phi_target, phi_wide, alpha=0.15, color='red', label='Blad (Epps-Pulley)')
ax.set_xlabel(r'$t$', fontsize=12)
ax.set_ylabel(r'$\varphi(t)$', fontsize=12)
ax.set_title('Funkcje charakterystyczne (SIGReg)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(-5, 5)

plt.tight_layout()
plt.savefig('characteristic_functions.pdf', bbox_inches='tight', dpi=300)
plt.savefig('characteristic_functions.png', bbox_inches='tight', dpi=200)
print("Plot 4 saved.")

print("\nAll plots generated successfully!")
