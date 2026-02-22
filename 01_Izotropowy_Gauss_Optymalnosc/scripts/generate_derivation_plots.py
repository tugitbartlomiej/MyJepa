"""Generate plots for 1D/2D/3D Gaussian derivation section."""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Ellipse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Plot 1: 1D Gaussian — varying mu and sigma
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

z = np.linspace(-6, 6, 500)

# Left: varying sigma
ax = axes[0]
for sigma, color, ls in [(0.5, '#F44336', '-'), (1.0, '#2196F3', '-'), (2.0, '#4CAF50', '-'), (3.0, '#FF9800', '--')]:
    pdf = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-z**2 / (2 * sigma**2))
    ax.plot(z, pdf, color=color, linewidth=2.2, linestyle=ls,
            label=rf'$\mu=0,\;\sigma^2={sigma**2:.1f}$')
    if sigma == 1.0:
        ax.fill_between(z, pdf, alpha=0.12, color=color)
ax.set_xlabel(r'$z$', fontsize=13)
ax.set_ylabel(r'$p(z)$', fontsize=13)
ax.set_title(r'1D Gauss: wpływ $\sigma^2$ (wariancja)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.85)
ax.axhline(0, color='gray', linewidth=0.5)

# Annotate parts of the formula on the sigma=1 curve
ax.annotate(r'$\frac{1}{\sqrt{2\pi}}$ (stała)', xy=(0, 0.399), xytext=(2.5, 0.6),
            fontsize=11, color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2196F3'))
ax.annotate(r'$e^{-z^2/2}$ maleje', xy=(2.0, 0.054), xytext=(3.5, 0.25),
            fontsize=11, color='#2196F3',
            arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#2196F3'))

# Right: varying mu
ax = axes[1]
for mu, color in [(-2, '#F44336'), (0, '#2196F3'), (1.5, '#4CAF50'), (3, '#FF9800')]:
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-(z - mu)**2 / 2)
    ax.plot(z, pdf, color=color, linewidth=2.2,
            label=rf'$\mu={mu},\;\sigma^2=1$')
ax.set_xlabel(r'$z$', fontsize=13)
ax.set_ylabel(r'$p(z)$', fontsize=13)
ax.set_title(r'1D Gauss: wpływ $\mu$ (średnia)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.5)
ax.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'gauss_1d.pdf'), bbox_inches='tight', dpi=300)
print("Plot 1 (1D) saved.")

# ============================================================
# Plot 2: 2D Gaussian — diagonal vs correlated vs isotropic
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x = np.linspace(-4, 4, 200)
y = np.linspace(-4, 4, 200)
X, Y = np.meshgrid(x, y)

def gauss2d(X, Y, mu, Sigma):
    Sinv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    dx = X - mu[0]
    dy = Y - mu[1]
    exponent = -(Sinv[0,0]*dx**2 + 2*Sinv[0,1]*dx*dy + Sinv[1,1]*dy**2) / 2
    return (1 / (2*np.pi*np.sqrt(det))) * np.exp(exponent)

configs = [
    (r'Izotropowy: $\Sigma = \mathbf{I}$' + '\n(LeJEPA!)',
     np.array([0,0]), np.array([[1,0],[0,1]]), 'Blues'),
    (r'Diagonalny: $\sigma_1^2 \neq \sigma_2^2$' + '\n(anizotropowy)',
     np.array([0,0]), np.array([[3,0],[0,0.5]]), 'Oranges'),
    (r'Ze korelacją: $\rho = 0.8$' + '\n(elementy pozadiagonalne)',
     np.array([0,0]), np.array([[1,0.8],[0.8,1]]), 'Greens'),
]

for ax, (title, mu, Sigma, cmap) in zip(axes, configs):
    Z = gauss2d(X, Y, mu, Sigma)
    contour = ax.contourf(X, Y, Z, levels=15, cmap=cmap, alpha=0.85)
    ax.contour(X, Y, Z, levels=6, colors='k', linewidths=0.5, alpha=0.4)

    # Draw eigenvectors
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    for i in range(2):
        ev = eigvecs[:, i] * np.sqrt(eigvals[i]) * 1.5
        ax.annotate('', xy=ev, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(r'$z_1$', fontsize=12)
    ax.set_ylabel(r'$z_2$', fontsize=12)

    # Show Sigma matrix (use simple notation, matplotlib can't render pmatrix)
    s_str = (rf'$\Sigma = [{Sigma[0,0]:.1f},\;{Sigma[0,1]:.1f}'
             rf';\;{Sigma[1,0]:.1f},\;{Sigma[1,1]:.1f}]$')
    ax.text(0, -3.3, s_str, fontsize=11, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'gauss_2d_types.pdf'), bbox_inches='tight', dpi=300)
print("Plot 2 (2D types) saved.")

# ============================================================
# Plot 3: 3D Gaussian surfaces — iso vs aniso
# ============================================================
fig = plt.figure(figsize=(14, 6))

x3 = np.linspace(-4, 4, 120)
y3 = np.linspace(-4, 4, 120)
X3, Y3 = np.meshgrid(x3, y3)

# Compute both densities
Z_iso = (1/(2*np.pi)) * np.exp(-0.5*(X3**2 + Y3**2))

Sigma_aniso = np.array([[3.0, 0.0], [0.0, 0.3]])
Sinv = np.linalg.inv(Sigma_aniso)
det_a = np.linalg.det(Sigma_aniso)
Z_aniso = (1/(2*np.pi*np.sqrt(det_a))) * np.exp(-0.5*(Sinv[0,0]*X3**2 + Sinv[1,1]*Y3**2))

# Common z-axis limit for fair comparison
z_max = max(Z_iso.max(), Z_aniso.max()) * 1.05

# Isotropic 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X3, Y3, Z_iso, cmap='Blues', alpha=0.9, edgecolor='none',
                 rstride=2, cstride=2)
# Contours on the floor to show circular shape
ax1.contour(X3, Y3, Z_iso, levels=8, zdir='z', offset=0, cmap='Blues', alpha=0.5)
ax1.set_title(r'Izotropowy $\mathcal{N}(\mathbf{0}, \mathbf{I}_2)$',
              fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel(r'$z_1$', fontsize=11)
ax1.set_ylabel(r'$z_2$', fontsize=11)
ax1.set_zlabel(r'$p(z_1,z_2)$', fontsize=11)
ax1.set_zlim(0, z_max)
ax1.view_init(elev=30, azim=-50)

# Anisotropic 3D surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X3, Y3, Z_aniso, cmap='Oranges', alpha=0.9, edgecolor='none',
                 rstride=2, cstride=2)
# Contours on the floor to show elliptical shape
ax2.contour(X3, Y3, Z_aniso, levels=8, zdir='z', offset=0, cmap='Oranges', alpha=0.5)
ax2.set_title(r'Anizotropowy: $\sigma_1^2=3, \sigma_2^2=0.3$',
              fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel(r'$z_1$', fontsize=11)
ax2.set_ylabel(r'$z_2$', fontsize=11)
ax2.set_zlabel(r'$p(z_1,z_2)$', fontsize=11)
ax2.set_zlim(0, z_max)
ax2.view_init(elev=30, azim=-50)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'gauss_3d_surfaces.pdf'), bbox_inches='tight', dpi=300)
print("Plot 3 (3D surfaces) saved.")

# ============================================================
# Plot 3b: TRUE 3D isosurfaces — spheres vs ellipsoids (z1, z2, z3)
# ============================================================
fig = plt.figure(figsize=(14, 6))

# Helper: parametric sphere/ellipsoid
u_s = np.linspace(0, 2 * np.pi, 60)
v_s = np.linspace(0, np.pi, 40)
U, V = np.meshgrid(u_s, v_s)

# Base unit sphere coordinates
Xs = np.sin(V) * np.cos(U)
Ys = np.sin(V) * np.sin(U)
Zs = np.cos(V)

# --- Left: Isotropic (spheres) ---
ax1 = fig.add_subplot(121, projection='3d')

# 1-sigma and 2-sigma spheres
for r, alpha_val, label in [(1.0, 0.4, r'$1\sigma$'), (2.0, 0.15, r'$2\sigma$')]:
    ax1.plot_surface(r*Xs, r*Ys, r*Zs, color='#2196F3', alpha=alpha_val,
                     edgecolor='#1565C0', linewidth=0.1, rstride=3, cstride=3)

# Axes through center
ax_len = 2.8
for vec, col, lbl in [([1,0,0], '#F44336', r'$z_1$'),
                       ([0,1,0], '#4CAF50', r'$z_2$'),
                       ([0,0,1], '#FF9800', r'$z_3$')]:
    v = np.array(vec) * ax_len
    ax1.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, linewidth=2.5)
    ax1.text(v[0]*1.15, v[1]*1.15, v[2]*1.15, lbl, fontsize=13,
             fontweight='bold', color=col)

ax1.set_title(r'Izotropowy $\boldsymbol{\Sigma} = \mathbf{I}_3$' + '\n(izopowierzchnie = sfery)',
              fontsize=12, fontweight='bold', pad=5)
ax1.set_xlim(-3, 3); ax1.set_ylim(-3, 3); ax1.set_zlim(-3, 3)
ax1.set_xlabel(r'$z_1$', fontsize=10)
ax1.set_ylabel(r'$z_2$', fontsize=10)
ax1.set_zlabel(r'$z_3$', fontsize=10)
ax1.view_init(elev=20, azim=-55)

# --- Right: Anisotropic (ellipsoids) ---
ax2 = fig.add_subplot(122, projection='3d')

# sigma values: sigma_1=sqrt(3)~1.73, sigma_2=sqrt(0.3)~0.55, sigma_3=1.0
sigmas = np.array([np.sqrt(3.0), np.sqrt(0.3), 1.0])

for scale, alpha_val, label in [(1.0, 0.4, r'$1\sigma$'), (2.0, 0.15, r'$2\sigma$')]:
    Xe = scale * sigmas[0] * Xs
    Ye = scale * sigmas[1] * Ys
    Ze = scale * sigmas[2] * Zs
    ax2.plot_surface(Xe, Ye, Ze, color='#FF9800', alpha=alpha_val,
                     edgecolor='#E65100', linewidth=0.1, rstride=3, cstride=3)

# Axes with lengths proportional to sigma
for vec, sig, col, lbl in [([1,0,0], sigmas[0], '#F44336', r'$z_1\;(\sigma_1\!=\!\sqrt{3})$'),
                             ([0,1,0], sigmas[1], '#4CAF50', r'$z_2\;(\sigma_2\!=\!\sqrt{0.3})$'),
                             ([0,0,1], sigmas[2], '#FF9800', r'$z_3\;(\sigma_3\!=\!1)$')]:
    v = np.array(vec) * ax_len
    ax2.plot([0, v[0]], [0, v[1]], [0, v[2]], color=col, linewidth=2.5)
    ax2.text(v[0]*1.15, v[1]*1.15, v[2]*1.15, lbl, fontsize=10,
             fontweight='bold', color=col)

ax2.set_title(r'Anizotropowy $\sigma_1^2\!=\!3,\;\sigma_2^2\!=\!0.3,\;\sigma_3^2\!=\!1$'
              + '\n(izopowierzchnie = elipsoidy)',
              fontsize=12, fontweight='bold', pad=5)
ax2.set_xlim(-4, 4); ax2.set_ylim(-4, 4); ax2.set_zlim(-3, 3)
ax2.set_xlabel(r'$z_1$', fontsize=10)
ax2.set_ylabel(r'$z_2$', fontsize=10)
ax2.set_zlabel(r'$z_3$', fontsize=10)
ax2.view_init(elev=20, azim=-55)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'gauss_3d_isosurfaces.pdf'), bbox_inches='tight', dpi=300)
print("Plot 3b (3D isosurfaces) saved.")

# ============================================================
# Plot 4: Formula decomposition visual
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(12, 5))

z_vals = np.linspace(-4, 4, 500)
pdf_standard = (1/np.sqrt(2*np.pi)) * np.exp(-z_vals**2 / 2)
exp_part = np.exp(-z_vals**2 / 2)
quadratic = z_vals**2 / 2

# Plot the decomposition
ax.plot(z_vals, pdf_standard, 'b-', linewidth=3, label=r'$p(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ (wynik)')
ax.plot(z_vals, exp_part, 'r--', linewidth=2, label=r'$e^{-z^2/2}$ (eksponenta)')
ax.plot(z_vals, quadratic, 'g-.', linewidth=2, label=r'$z^2/2$ (kwadrat odległości)')
ax.axhline(1/np.sqrt(2*np.pi), color='purple', linewidth=1, linestyle=':',
           label=rf'$1/\sqrt{{2\pi}} \approx {1/np.sqrt(2*np.pi):.3f}$ (stała)')

ax.fill_between(z_vals, pdf_standard, alpha=0.1, color='blue')
ax.set_xlabel(r'$z$', fontsize=13)
ax.set_ylabel('Wartość', fontsize=13)
ax.set_title('Rozkład 1D Gaussa: z czego się składa?', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.set_xlim(-4, 4)
ax.set_ylim(0, 2.5)
ax.axhline(0, color='gray', linewidth=0.5)

# Add annotations
ax.annotate(r'Maksimum w $z=0$:' + '\n' + r'$p(0) = 1/\sqrt{2\pi}$',
            xy=(0, 0.399), xytext=(-3.5, 1.5), fontsize=11,
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.annotate(r'$z^2/2$ rośnie' + '\n' + r'$\Rightarrow e^{-z^2/2}$ maleje' + '\n' + r'$\Rightarrow p(z)$ maleje',
            xy=(2.5, 0.018), xytext=(2.5, 1.5), fontsize=11,
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'gauss_decomposition.pdf'), bbox_inches='tight', dpi=300)
print("Plot 4 (decomposition) saved.")

# ============================================================
# Plot 5: Isolines comparison 2D — circles vs ellipses
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

titles = [
    r'$\Sigma = \mathbf{I}$: okręgi (izotropowy)',
    r'$\Sigma = \mathrm{diag}(3, 0.5)$: elipsy (anizotrop.)',
    r'$\Sigma$ z $\rho=0.7$: obrócone elipsy',
]
sigmas = [
    np.array([[1, 0], [0, 1]]),
    np.array([[3, 0], [0, 0.5]]),
    np.array([[1, 0.7], [0.7, 1]]),
]
cmaps = ['Blues', 'Oranges', 'Greens']

for ax, title, Sigma, cmap in zip(axes, titles, sigmas, cmaps):
    Z = gauss2d(X, Y, np.array([0,0]), Sigma)
    ax.contour(X, Y, Z, levels=8, cmap=cmap, linewidths=2)

    # Mark eigenvalues
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    for i in range(2):
        ev = eigvecs[:, i] * np.sqrt(eigvals[i]) * 1.5
        ax.annotate('', xy=ev, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
        # Label
        ax.text(ev[0]*1.15 + 0.1, ev[1]*1.15 + 0.1,
                rf'$\lambda_{i+1}={eigvals[i]:.2f}$',
                fontsize=10, color='red', fontweight='bold')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'$z_1$', fontsize=12)
    ax.set_ylabel(r'$z_2$', fontsize=12)
    ax.axhline(0, color='gray', linewidth=0.3)
    ax.axvline(0, color='gray', linewidth=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'gauss_isolines.pdf'), bbox_inches='tight', dpi=300)
print("Plot 5 (isolines) saved.")

print("\nAll derivation plots generated!")
