#!/usr/bin/env python3
"""
Generate projection visualization plots for SIGReg explanation.
Shows how u = a^T z works: projecting K-dimensional embeddings onto 1D.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "latex", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Style ---
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

np.random.seed(42)


def plot1_simple_projection():
    """
    Plot 1: Simple 2D example of projection u = a^T z.
    Shows N=8 embedding points in 2D, a direction vector a,
    and the projections of each point onto that direction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Generate some 2D "embeddings" (isotropic Gaussian)
    N = 8
    Z = np.random.randn(N, 2) * 0.9

    # Direction vector a (unit vector at ~35 degrees)
    angle_deg = 35
    angle = np.radians(angle_deg)
    a = np.array([np.cos(angle), np.sin(angle)])

    # Compute projections u = a^T z
    U = Z @ a  # shape (N,)
    # Projected points on the line
    proj_points = np.outer(U, a)  # shape (N, 2)

    # --- Panel 1: Just the embeddings ---
    ax = axes[0]
    ax.set_title(r"$\mathbf{z}$ — embeddingi (K=2)", fontsize=14, fontweight='bold')
    for i in range(N):
        ax.plot(Z[i, 0], Z[i, 1], 'o', color='#2196F3', markersize=10, zorder=5)
        ax.annotate(rf'$\mathbf{{z}}_{i+1}$',
                    (Z[i, 0], Z[i, 1]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=10, color='#1565C0')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Embeddings + direction + projection lines ---
    ax = axes[1]
    ax.set_title(r"Rzut: $u = \mathbf{a}^\top \mathbf{z}$", fontsize=14, fontweight='bold')

    # Draw direction line (extended)
    line_t = np.linspace(-2.5, 2.5, 100)
    line_x = line_t * a[0]
    line_y = line_t * a[1]
    ax.plot(line_x, line_y, '-', color='#E53935', linewidth=2.5, alpha=0.7,
            label=f'kierunek $\\mathbf{{a}}$')

    # Draw direction vector a (thick arrow)
    ax.annotate('', xy=(a[0]*1.5, a[1]*1.5), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=3))
    ax.annotate(r'$\mathbf{a}$', xy=(a[0]*1.6, a[1]*1.6),
                fontsize=14, fontweight='bold', color='#B71C1C')

    # Draw embedding points, projection lines, and projected points
    for i in range(N):
        # Original point
        ax.plot(Z[i, 0], Z[i, 1], 'o', color='#2196F3', markersize=10, zorder=5)
        # Projected point on the line
        ax.plot(proj_points[i, 0], proj_points[i, 1], 's', color='#FF9800',
                markersize=8, zorder=5)
        # Dashed line connecting original to projection (perpendicular)
        ax.plot([Z[i, 0], proj_points[i, 0]], [Z[i, 1], proj_points[i, 1]],
                '--', color='gray', linewidth=1, alpha=0.6)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Legend
    blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3',
                          markersize=10, label=r'$\mathbf{z}_j$ (embeddingi)')
    orange_sq = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF9800',
                           markersize=8, label=r'$u_j \cdot \mathbf{a}$ (rzuty)')
    red_line = plt.Line2D([0], [0], color='#E53935', linewidth=2.5,
                          label=r'kierunek $\mathbf{a}$')
    ax.legend(handles=[blue_dot, orange_sq, red_line], loc='lower right', fontsize=9)

    # --- Panel 3: The resulting 1D values (histogram) ---
    ax = axes[2]
    ax.set_title(r"$u_1, u_2, \ldots, u_N$ — wartości 1D", fontsize=14, fontweight='bold')

    # Sort for display
    U_sorted_idx = np.argsort(U)

    # Draw number line
    ax.axhline(0.5, color='gray', linewidth=1)

    # Place u values on number line
    for i in range(N):
        idx = i
        ax.plot(U[idx], 0.5, 's', color='#FF9800', markersize=12, zorder=5)
        ax.annotate(rf'$u_{idx+1}={U[idx]:.1f}$',
                    (U[idx], 0.5),
                    textcoords="offset points",
                    xytext=(0, 15 if idx % 2 == 0 else -20),
                    fontsize=9, ha='center', color='#E65100')

    # Add histogram below
    ax.hist(U, bins=6, density=True, alpha=0.4, color='#FF9800',
            edgecolor='#E65100', bottom=-0.6, orientation='vertical')

    # Overlay Gaussian N(0,1) curve
    x_gauss = np.linspace(-3, 3, 200)
    y_gauss = (1/np.sqrt(2*np.pi)) * np.exp(-x_gauss**2 / 2)
    ax.plot(x_gauss, y_gauss - 0.6, 'k-', linewidth=1.5, alpha=0.5,
            label=r'$\mathcal{N}(0,1)$ cel')

    ax.set_xlabel(r'$u = \mathbf{a}^\top \mathbf{z}$')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 1.3)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "projection_explanation.pdf"))
    plt.close()
    print("Saved: projection_explanation.pdf")


def plot2_multiple_directions():
    """
    Plot 2: Same point cloud, 3 different projection directions.
    Shows that different directions give different 1D distributions.
    Isotropic => all directions look like N(0,1).
    Anisotropic => different directions give different variances.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # --- Row 1: ISOTROPIC embeddings ---
    Z_iso = np.random.randn(50, 2) * 1.0

    # 3 directions
    angles_deg = [0, 60, 120]
    colors_a = ['#E53935', '#43A047', '#1E88E5']

    ax = axes[0, 0]
    ax.set_title("Izotropowy: embeddingi", fontsize=13, fontweight='bold')
    ax.scatter(Z_iso[:, 0], Z_iso[:, 1], c='#2196F3', s=30, alpha=0.7, zorder=5)
    for k, ang_deg in enumerate(angles_deg):
        ang = np.radians(ang_deg)
        a_vec = np.array([np.cos(ang), np.sin(ang)])
        # Draw direction
        ax.annotate('', xy=(a_vec[0]*2.5, a_vec[1]*2.5), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=colors_a[k], lw=2.5))
        ax.annotate(rf'$\mathbf{{a}}_{k+1}$',
                    xy=(a_vec[0]*2.7, a_vec[1]*2.7),
                    fontsize=12, fontweight='bold', color=colors_a[k])
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    ax.grid(True, alpha=0.3)

    for k, ang_deg in enumerate(angles_deg):
        ang = np.radians(ang_deg)
        a_vec = np.array([np.cos(ang), np.sin(ang)])
        U = Z_iso @ a_vec

        ax = axes[0, k+1]
        ax.set_title(rf"$\mathbf{{a}}_{k+1}$ ({ang_deg}°): $\sigma^2 \approx {np.var(U):.2f}$",
                     fontsize=12, color=colors_a[k], fontweight='bold')
        ax.hist(U, bins=12, density=True, alpha=0.5, color=colors_a[k],
                edgecolor=colors_a[k])
        x_g = np.linspace(-3.5, 3.5, 200)
        y_g = (1/np.sqrt(2*np.pi)) * np.exp(-x_g**2 / 2)
        ax.plot(x_g, y_g, 'k--', linewidth=1.5, alpha=0.7, label=r'$\mathcal{N}(0,1)$')
        ax.set_xlabel(rf'$u = \mathbf{{a}}_{k+1}^\top \mathbf{{z}}$')
        ax.set_ylabel('gęstość')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(0, 0.55)
        ax.legend(fontsize=9)

    # --- Row 2: ANISOTROPIC embeddings ---
    # Stretch along z1 axis (variance 4 in z1, variance 0.25 in z2)
    Z_aniso = np.random.randn(50, 2) @ np.diag([2.0, 0.5])

    ax = axes[1, 0]
    ax.set_title("Anizotropowy: embeddingi", fontsize=13, fontweight='bold')
    ax.scatter(Z_aniso[:, 0], Z_aniso[:, 1], c='#2196F3', s=30, alpha=0.7, zorder=5)
    for k, ang_deg in enumerate(angles_deg):
        ang = np.radians(ang_deg)
        a_vec = np.array([np.cos(ang), np.sin(ang)])
        ax.annotate('', xy=(a_vec[0]*2.5, a_vec[1]*2.5), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=colors_a[k], lw=2.5))
        ax.annotate(rf'$\mathbf{{a}}_{k+1}$',
                    xy=(a_vec[0]*2.7, a_vec[1]*2.7),
                    fontsize=12, fontweight='bold', color=colors_a[k])
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$z_1$')
    ax.set_ylabel(r'$z_2$')
    ax.grid(True, alpha=0.3)

    for k, ang_deg in enumerate(angles_deg):
        ang = np.radians(ang_deg)
        a_vec = np.array([np.cos(ang), np.sin(ang)])
        U = Z_aniso @ a_vec

        ax = axes[1, k+1]
        ax.set_title(rf"$\mathbf{{a}}_{k+1}$ ({ang_deg}°): $\sigma^2 \approx {np.var(U):.2f}$",
                     fontsize=12, color=colors_a[k], fontweight='bold')
        ax.hist(U, bins=12, density=True, alpha=0.5, color=colors_a[k],
                edgecolor=colors_a[k])
        x_g = np.linspace(-5, 5, 200)
        y_g = (1/np.sqrt(2*np.pi)) * np.exp(-x_g**2 / 2)
        ax.plot(x_g, y_g, 'k--', linewidth=1.5, alpha=0.7, label=r'$\mathcal{N}(0,1)$')
        ax.set_xlabel(rf'$u = \mathbf{{a}}_{k+1}^\top \mathbf{{z}}$')
        ax.set_ylabel('gęstość')
        ax.set_xlim(-5, 5)
        ax.set_ylim(0, 0.85)
        ax.legend(fontsize=9)

    # Row labels
    fig.text(0.01, 0.72, 'Izotropowy\n(cel)', fontsize=14, fontweight='bold',
             color='#43A047', ha='left', va='center', rotation=90)
    fig.text(0.01, 0.28, 'Anizotropowy\n(problem)', fontsize=14, fontweight='bold',
             color='#E53935', ha='left', va='center', rotation=90)

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig(os.path.join(FIGURES_DIR, "projection_directions.pdf"))
    plt.close()
    print("Saved: projection_directions.pdf")


def plot3_concrete_example():
    """
    Plot 3: Concrete numerical example with K=3.
    Shows the dot product step by step.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.97, r'Przykład: $u = \mathbf{a}^\top \mathbf{z}$ dla $K=3$',
            fontsize=18, fontweight='bold', ha='center', va='top',
            transform=ax.transAxes)

    # The vectors
    z_vals = [0.5, -1.2, 0.8]
    a_val = 1/np.sqrt(3)

    # Step 1: Define vectors
    y_start = 0.85
    ax.text(0.05, y_start,
            r'$\mathbf{z} = (0.5,\; -1.2,\; 0.8)$'
            r'  $\leftarrow$ embedding (wyjście enkodera, $K=3$ wymiary)',
            fontsize=15, va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0'))

    ax.text(0.05, y_start - 0.11,
            r'$\mathbf{a} = (0.577,\; 0.577,\; 0.577)$'
            r'  $\leftarrow$ losowy kierunek jednostkowy ($\|\mathbf{a}\| = 1$)',
            fontsize=15, va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828'))

    ax.text(0.15, y_start - 0.19,
            r'($= 1/\sqrt{3}$ w każdym wymiarze; $\sqrt{3 \cdot (1/\sqrt{3})^2} = 1$ )',
            fontsize=11, va='center', transform=ax.transAxes, color='#888')

    # Step 2: Dot product formula
    ax.text(0.05, y_start - 0.30,
            r'$u = \mathbf{a}^\top \mathbf{z} = a_1 \cdot z_1 + a_2 \cdot z_2 + a_3 \cdot z_3$',
            fontsize=15, va='center', transform=ax.transAxes,
            fontweight='bold')

    # Step 3: Numbers substituted
    ax.text(0.05, y_start - 0.41,
            r'$= 0.577 \cdot 0.5 \;+\; 0.577 \cdot (-1.2) \;+\; 0.577 \cdot 0.8$',
            fontsize=15, va='center', transform=ax.transAxes)

    # Step 4: Intermediate
    v1 = a_val * z_vals[0]
    v2 = a_val * z_vals[1]
    v3 = a_val * z_vals[2]
    u_val = v1 + v2 + v3

    ax.text(0.05, y_start - 0.51,
            rf'$= {v1:.3f} + ({v2:.3f}) + {v3:.3f}$',
            fontsize=15, va='center', transform=ax.transAxes)

    ax.text(0.05, y_start - 0.62,
            rf'$u = {u_val:.3f}$',
            fontsize=18, fontweight='bold', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100'))

    # Annotation
    ax.text(0.40, y_start - 0.62,
            r'$\leftarrow$ jedna liczba! (z $K=3$ wymiarów zrobiliśmy 1 wymiar)',
            fontsize=14, va='center', transform=ax.transAxes, color='#E65100')

    # Bottom box: meaning
    ax.text(0.5, 0.08,
            r"$\mathbf{z}$ — wektor embeddingu ($K$ liczb)       "
            r"$\mathbf{a}$ — losowy kierunek ($K$ liczb, $\|\mathbf{a}\|=1$)       "
            r"$u$ — iloczyn skalarny = 1 liczba",
            fontsize=12, ha='center', va='center',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32'))

    plt.savefig(os.path.join(FIGURES_DIR, "projection_example.pdf"))
    plt.close()
    print("Saved: projection_example.pdf")


def plot4_batch_pipeline():
    """
    Plot 4: Full pipeline diagram:
    Batch of images -> encoder -> embeddings z_1..z_N -> random a -> u_1..u_N -> EP test
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.axis('off')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)

    # Boxes
    box_style = dict(boxstyle='round,pad=0.4', linewidth=2)

    # Box 1: Images
    ax.text(1.5, 2.5, r'Batch obrazów' + '\n' + r'$\mathbf{x}_1, \ldots, \mathbf{x}_N$',
            fontsize=12, ha='center', va='center',
            bbox=dict(**box_style, facecolor='#E3F2FD', edgecolor='#1565C0'))

    # Arrow 1
    ax.annotate('', xy=(3.2, 2.5), xytext=(2.6, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(2.9, 2.9, r'$f_\theta$', fontsize=11, ha='center', color='gray')

    # Box 2: Embeddings
    ax.text(4.8, 2.5, r'Embeddingi' + '\n' +
            r'$\mathbf{z}_1, \ldots, \mathbf{z}_N$' + '\n' + r'(K wymiarów)',
            fontsize=12, ha='center', va='center',
            bbox=dict(**box_style, facecolor='#E8F5E9', edgecolor='#2E7D32'))

    # Arrow 2
    ax.annotate('', xy=(7.0, 2.5), xytext=(6.3, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(6.65, 2.9, r'$\times\,\mathbf{a}$', fontsize=12, ha='center', color='#E53935')

    # Direction vector
    ax.text(6.65, 3.7, r'Losowy' + '\n' + r'kierunek $\mathbf{a}$',
            fontsize=10, ha='center', va='center', color='#E53935',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#E53935', linewidth=1))
    ax.annotate('', xy=(6.65, 3.15), xytext=(6.65, 3.45),
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))

    # Box 3: Projections
    ax.text(8.7, 2.5, r'Rzuty 1D' + '\n' +
            r'$u_1, \ldots, u_N$' + '\n' + r'(1 wymiar!)',
            fontsize=12, ha='center', va='center',
            bbox=dict(**box_style, facecolor='#FFF3E0', edgecolor='#E65100'))

    # Arrow 3
    ax.annotate('', xy=(10.8, 2.5), xytext=(10.1, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(10.45, 2.9, 'ECF', fontsize=11, ha='center', color='gray')

    # Box 4: EP test
    ax.text(12.5, 2.5, r'Test EP' + '\n' + r'$\hat{\varphi}$ vs $e^{-t^2/2}$',
            fontsize=12, ha='center', va='center',
            bbox=dict(**box_style, facecolor='#F3E5F5', edgecolor='#7B1FA2'))

    # Arrow 4
    ax.annotate('', xy=(14.3, 2.5), xytext=(13.7, 2.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Box 5: Loss
    ax.text(15.2, 2.5, r'$\mathcal{L}_{\mathrm{SIGReg}}$',
            fontsize=16, ha='center', va='center', fontweight='bold',
            bbox=dict(**box_style, facecolor='#FCE4EC', edgecolor='#C62828'))

    # Bottom: dimension annotations
    ax.text(1.5, 0.8, r'$[B, C, H, W]$', fontsize=10, ha='center', color='#666')
    ax.text(4.8, 0.8, r'$[N, K]$', fontsize=10, ha='center', color='#666')
    ax.text(8.7, 0.8, r'$[N]$', fontsize=10, ha='center', color='#666')
    ax.text(12.5, 0.8, r'skalar', fontsize=10, ha='center', color='#666')
    ax.text(15.2, 0.8, r'skalar', fontsize=10, ha='center', color='#666')

    # Title
    ax.text(8, 4.7, r'Pipeline: od obrazów do SIGReg loss',
            fontsize=16, ha='center', va='center', fontweight='bold')

    # u = a^T z annotation
    ax.text(6.65, 1.4, r'$u_j = \mathbf{a}^\top \mathbf{z}_j$',
            fontsize=13, ha='center', va='center', color='#E53935',
            fontstyle='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E53935', alpha=0.8))

    plt.savefig(os.path.join(FIGURES_DIR, "projection_pipeline.pdf"))
    plt.close()
    print("Saved: projection_pipeline.pdf")


if __name__ == "__main__":
    print("Generating projection visualization plots...")
    plot1_simple_projection()
    plot2_multiple_directions()
    plot3_concrete_example()
    plot4_batch_pipeline()
    print("All plots generated!")
