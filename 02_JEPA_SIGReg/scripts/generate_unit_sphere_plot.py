#!/usr/bin/env python3
"""
Visualize the unit sphere S^{K-1} and random direction vectors a.
Shows: 2D circle (S^1) and 3D sphere (S^2) with random unit vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "latex", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

np.random.seed(42)


def random_unit_vectors(K, M):
    """Generate M random unit vectors in R^K."""
    v = np.random.randn(M, K)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def plot_unit_sphere():
    fig = plt.figure(figsize=(18, 7.5))

    # =============================================
    # Panel 1: K=2, unit circle S^1
    # =============================================
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title(r'$K=2$:  sfera $\mathbb{S}^1$ = okrąg', fontsize=15, fontweight='bold')

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax1.plot(np.cos(theta), np.sin(theta), '-', color='#BBBBBB', linewidth=2.5, zorder=1)
    ax1.fill(np.cos(theta), np.sin(theta), alpha=0.04, color='#2196F3')

    # Draw coordinate axes
    ax1.axhline(0, color='#DDDDDD', linewidth=0.8, zorder=0)
    ax1.axvline(0, color='#DDDDDD', linewidth=0.8, zorder=0)

    # Random direction vectors
    M = 6
    a_vecs = random_unit_vectors(2, M)
    colors = ['#E53935', '#43A047', '#1E88E5', '#FF9800', '#9C27B0', '#00ACC1']

    for i in range(M):
        ax1.annotate('', xy=(a_vecs[i, 0], a_vecs[i, 1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=colors[i], lw=2.5),
                     zorder=5)
        # Dot at the tip (on the circle)
        ax1.plot(a_vecs[i, 0], a_vecs[i, 1], 'o', color=colors[i],
                 markersize=8, zorder=6)
        # Label
        offset_x = a_vecs[i, 0] * 0.18
        offset_y = a_vecs[i, 1] * 0.18
        ax1.text(a_vecs[i, 0] + offset_x, a_vecs[i, 1] + offset_y,
                 rf'$\mathbf{{a}}_{i+1}$', fontsize=11, fontweight='bold',
                 color=colors[i], ha='center', va='center', zorder=7)

    # Origin
    ax1.plot(0, 0, 'ko', markersize=5, zorder=6)
    ax1.text(0.08, -0.12, r'$\mathbf{0}$', fontsize=12, color='black')

    # Annotation: ||a|| = 1
    ax1.annotate(r'$\|\mathbf{a}\| = 1$', xy=(0.707, 0.707),
                 xytext=(1.15, 1.25), fontsize=13, color='#555',
                 arrowprops=dict(arrowstyle='->', color='#999', lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4',
                           edgecolor='#F9A825'))

    ax1.set_xlim(-1.6, 1.6)
    ax1.set_ylim(-1.6, 1.6)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$a_1$', fontsize=13)
    ax1.set_ylabel(r'$a_2$', fontsize=13)
    ax1.grid(True, alpha=0.2)

    # Bottom text
    ax1.text(0, -1.45, r'Każda strzałka to jeden kierunek $\mathbf{a}$' + '\n'
             r'Wszystkie mają długość 1 (kończą się na okręgu)',
             fontsize=10, ha='center', va='top', color='#555',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#DDD', alpha=0.9))

    # =============================================
    # Panel 2: K=3, unit sphere S^2
    # =============================================
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.set_title(r'$K=3$:  sfera $\mathbb{S}^2$ = kula', fontsize=15, fontweight='bold',
                  pad=15)

    # Draw transparent sphere
    u_sphere = np.linspace(0, 2*np.pi, 40)
    v_sphere = np.linspace(0, np.pi, 25)
    x_s = np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_s = np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_s = np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
    ax2.plot_surface(x_s, y_s, z_s, alpha=0.08, color='#2196F3',
                     edgecolor='#BBBBBB', linewidth=0.15)

    # Draw great circles for reference
    circle_t = np.linspace(0, 2*np.pi, 100)
    # XY plane
    ax2.plot(np.cos(circle_t), np.sin(circle_t), np.zeros_like(circle_t),
             '-', color='#CCCCCC', linewidth=0.8, alpha=0.6)
    # XZ plane
    ax2.plot(np.cos(circle_t), np.zeros_like(circle_t), np.sin(circle_t),
             '-', color='#CCCCCC', linewidth=0.8, alpha=0.6)
    # YZ plane
    ax2.plot(np.zeros_like(circle_t), np.cos(circle_t), np.sin(circle_t),
             '-', color='#CCCCCC', linewidth=0.8, alpha=0.6)

    # Random direction vectors in 3D
    M_3d = 8
    a_vecs_3d = random_unit_vectors(3, M_3d)
    colors_3d = ['#E53935', '#43A047', '#1E88E5', '#FF9800',
                 '#9C27B0', '#00ACC1', '#F44336', '#8BC34A']

    for i in range(M_3d):
        ax2.quiver(0, 0, 0,
                   a_vecs_3d[i, 0], a_vecs_3d[i, 1], a_vecs_3d[i, 2],
                   color=colors_3d[i], arrow_length_ratio=0.12, linewidth=2.2)
        ax2.scatter(*a_vecs_3d[i], color=colors_3d[i], s=40, zorder=5)

    # Origin
    ax2.scatter(0, 0, 0, color='black', s=30, zorder=6)

    ax2.set_xlabel(r'$a_1$', fontsize=12, labelpad=5)
    ax2.set_ylabel(r'$a_2$', fontsize=12, labelpad=5)
    ax2.set_zlabel(r'$a_3$', fontsize=12, labelpad=5)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_zlim(-1.2, 1.2)
    ax2.view_init(elev=20, azim=135)

    # =============================================
    # Panel 3: Explanation / how to generate
    # =============================================
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    ax3.text(5, 9.5, r'Co to jest $\mathbb{S}^{K-1}$?', fontsize=16,
             fontweight='bold', ha='center', va='top')

    explanation = [
        (8.5, r'$\mathbb{S}^{K-1} = \{\mathbf{a} \in \mathbb{R}^K : \|\mathbf{a}\| = 1\}$',
         14, 'black', '#E3F2FD', '#1565C0'),
        (7.5, 'Zbiór wszystkich wektorów o długości 1', 12, '#555', None, None),
        (6.4, r'$K=2$:  $\mathbb{S}^1$ = okrąg  (obwód koła)', 13, '#E53935', None, None),
        (5.7, r'$K=3$:  $\mathbb{S}^2$ = sfera  (powierzchnia kuli)', 13, '#43A047', None, None),
        (5.0, r'$K=128$:  $\mathbb{S}^{127}$ = hipersfera 128D', 13, '#1E88E5', None, None),
    ]

    for y, txt, fs, col, bg, ec in explanation:
        kwargs = dict(fontsize=fs, ha='center', va='center', color=col)
        if bg:
            kwargs['bbox'] = dict(boxstyle='round,pad=0.4', facecolor=bg, edgecolor=ec)
        ax3.text(5, y, txt, **kwargs)

    # How to generate
    ax3.text(5, 3.8, 'Jak losujemy kierunek $\\mathbf{a}$?', fontsize=14,
             fontweight='bold', ha='center', va='center')

    steps = [
        (3.1, r'1.  Losujemy $K$ liczb z $\mathcal{N}(0,1)$:  $\mathbf{v} = (v_1, \ldots, v_K)$'),
        (2.4, r'2.  Obliczamy normę: $\|\mathbf{v}\| = \sqrt{v_1^2 + \cdots + v_K^2}$'),
        (1.7, r'3.  Normalizujemy: $\mathbf{a} = \mathbf{v} / \|\mathbf{v}\|$'),
        (0.9, r'Teraz $\|\mathbf{a}\| = 1$ $\checkmark$ i kierunek jest losowy równomiernie'),
    ]

    for y, txt in steps:
        col = '#333' if y > 1.0 else '#43A047'
        ax3.text(0.5, y, txt, fontsize=11.5, ha='left', va='center', color=col)

    # Box around steps
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.2, 0.5), 9.6, 3.7, boxstyle='round,pad=0.2',
                          facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1.2)
    ax3.add_patch(box)
    box.set_zorder(0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "unit_sphere.pdf"))
    plt.close()
    print("Saved: unit_sphere.pdf")


def plot_sphere_intuition_2d():
    """
    Extra 2D plot: show that ALL unit vectors lie on the circle,
    and the projection u = a^T z is the shadow on that direction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # --- Panel 1: What IS NOT on the sphere ---
    ax = axes[0]
    ax.set_title('Które wektory leżą na sferze?', fontsize=14, fontweight='bold')

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '-', color='#43A047', linewidth=3,
            label=r'$\mathbb{S}^1$ (okrąg $\|\mathbf{a}\|=1$)', zorder=2)
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.05, color='#43A047')

    # Good vectors (on circle)
    good = [(1, 0), (0, 1), (-1, 0), (0.707, 0.707), (-0.6, 0.8)]
    for i, (x, y) in enumerate(good):
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#43A047', lw=2.2))
        ax.plot(x, y, 'o', color='#43A047', markersize=8, zorder=5)

    # Bad vectors (NOT on circle - wrong length)
    bad = [(0.4, 0.3), (1.5, 0.8), (-0.2, -0.15), (0.9, 1.3)]
    for x, y in bad:
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.8,
                                   linestyle='dashed'))
        ax.plot(x, y, 'x', color='#E53935', markersize=10, markeredgewidth=2.5, zorder=5)
        norm = np.sqrt(x**2 + y**2)
        ax.text(x + 0.08, y + 0.08, rf'$\|.\|={norm:.1f}$',
                fontsize=9, color='#E53935')

    ax.plot(0, 0, 'ko', markersize=5, zorder=6)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$a_1$', fontsize=13)
    ax.set_ylabel(r'$a_2$', fontsize=13)
    ax.grid(True, alpha=0.2)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#43A047',
               markersize=10, label=r'$\|\mathbf{a}\|=1$ (na sferze)'),
        Line2D([0], [0], marker='x', color='#E53935', markeredgewidth=2.5,
               markersize=10, linestyle='None', label=r'$\|\mathbf{a}\|\neq 1$ (poza sferą)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11)

    # --- Panel 2: Meaning of K-1 ---
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(5, 9.5, r'Dlaczego $\mathbb{S}^{K\!-\!1}$, a nie $\mathbb{S}^K$?',
            fontsize=16, fontweight='bold', ha='center', va='top')

    lines = [
        (8.3, r'Indeks mówi o \textbf{wymiarowości} sfery, nie przestrzeni:', 13, '#333'),
        (7.2, r'$\mathbb{S}^1$ = okrąg = krzywa 1-wymiarowa', 14, '#E53935'),
        (6.6, r'(żyje w $\mathbb{R}^2$, ale sam jest 1D)', 11, '#888'),
        (5.5, r'$\mathbb{S}^2$ = sfera = powierzchnia 2-wymiarowa', 14, '#43A047'),
        (4.9, r'(żyje w $\mathbb{R}^3$, ale sama jest 2D)', 11, '#888'),
        (3.8, r'$\mathbb{S}^{127}$ = hipersfera 127-wymiarowa', 14, '#1E88E5'),
        (3.2, r'(żyje w $\mathbb{R}^{128}$, ale sama jest 127D)', 11, '#888'),
    ]

    for y, txt, fs, col in lines:
        ax.text(5, y, txt, fontsize=fs, ha='center', va='center', color=col)

    # Key insight box
    ax.text(5, 1.5,
            r'Ogólnie: $\mathbb{S}^{K-1} \subset \mathbb{R}^K$' + '\n'
            r'warunek $\|\mathbf{a}\|=1$ "zabiera" 1 stopień swobody' + '\n'
            r'więc sfera w $\mathbb{R}^K$ ma wymiar $K\!-\!1$',
            fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32'),
            linespacing=1.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "unit_sphere_intuition.pdf"))
    plt.close()
    print("Saved: unit_sphere_intuition.pdf")


if __name__ == "__main__":
    print("Generating unit sphere visualizations...")
    plot_unit_sphere()
    plot_sphere_intuition_2d()
    print("All done!")
