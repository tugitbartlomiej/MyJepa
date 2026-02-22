"""Generate plots explaining Vision Transformer (ViT) architecture."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# Plot 1: Image → Patches → Flat vectors
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

# Panel 1: Original image (simulated)
ax = axes[0]
np.random.seed(7)
img = np.random.rand(224, 224, 3) * 0.3 + 0.3
# Add some structure
for i in range(0, 224, 32):
    for j in range(0, 224, 32):
        img[i:i+32, j:j+32] *= (0.5 + np.random.rand() * 0.8)
img = np.clip(img, 0, 1)
ax.imshow(img)
ax.set_title(r'1. Obraz wejściowy' + '\n' + r'$224 \times 224 \times 3$ (RGB)', fontsize=12, fontweight='bold')
ax.set_xlabel('piksele', fontsize=10)
ax.set_ylabel('piksele', fontsize=10)

# Panel 2: Image with patch grid
ax = axes[1]
ax.imshow(img)
patch_size = 16
n_patches = 224 // patch_size  # 14
for i in range(n_patches + 1):
    ax.axhline(i * patch_size, color='red', linewidth=1.0, alpha=0.7)
    ax.axvline(i * patch_size, color='red', linewidth=1.0, alpha=0.7)
# Highlight one patch
rect = Rectangle((3*patch_size, 2*patch_size), patch_size, patch_size,
                  linewidth=3, edgecolor='yellow', facecolor='yellow', alpha=0.3)
ax.add_patch(rect)
ax.set_title(r'2. Podziel na patche' + '\n' + rf'$14 \times 14 = {n_patches**2}$ patchy ({patch_size}x{patch_size})',
             fontsize=12, fontweight='bold')
ax.set_xlabel(f'siatka {n_patches}x{n_patches}', fontsize=10)

# Panel 3: Patches as flat vectors
ax = axes[2]
n_show = 14
colors = plt.cm.tab20(np.linspace(0, 1, n_show))
for i in range(n_show):
    bar_height = 0.6
    y = n_show - 1 - i
    ax.barh(y, 768, height=bar_height, color=colors[i], alpha=0.7, edgecolor='k', linewidth=0.5)
    ax.text(768/2, y, rf'patch {i+1}: wektor $\in \mathbb{{R}}^{{768}}$',
            ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(768/2, n_show + 0.3, r'$16 \times 16 \times 3 = 768$ wymiarów', fontsize=10,
        ha='center', fontweight='bold', color='#B71C1C')
ax.set_xlim(0, 900)
ax.set_ylim(-0.5, n_show + 1)
ax.set_yticks([])
ax.set_xlabel('wymiary wektora', fontsize=10)
ax.set_title(r'3. Każdy patch → wektor' + '\n' + r'(spłaszczenie pikseli)', fontsize=12, fontweight='bold')

# Panel 4: After linear projection
ax = axes[3]
# CLS token
ax.barh(n_show, 384, height=bar_height, color='gold', edgecolor='k', linewidth=1.5)
ax.text(384/2, n_show, r'[CLS] token', ha='center', va='center', fontsize=9, fontweight='bold')
for i in range(n_show):
    y = n_show - 1 - i
    ax.barh(y, 384, height=bar_height, color=colors[i], alpha=0.7, edgecolor='k', linewidth=0.5)
    ax.text(384/2, y, rf'patch {i+1}: $\mathbb{{R}}^{{384}}$',
            ha='center', va='center', fontsize=8)
ax.text(384/2, n_show + 1.0, r'Projekcja liniowa: $768 \to 384$' + '\n' + '+ token [CLS] na górze',
        fontsize=10, ha='center', fontweight='bold', color='#1565C0')
ax.set_xlim(0, 500)
ax.set_ylim(-0.5, n_show + 1.8)
ax.set_yticks([])
ax.set_xlabel('wymiary embeddingu', fontsize=10)
ax.set_title(r'4. Projekcja + [CLS] token' + '\n' + r'$197$ tokenów $\times$ $384$ dim', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'vit_patches.pdf'), bbox_inches='tight', dpi=300)
print("Plot 1 (patches) saved.")

# ============================================================
# Plot 2: Full ViT pipeline diagram
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color, fontsize=10, text_color='black'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

# Title
ax.text(8, 9.6, 'Vision Transformer (ViT-Small) — pełny pipeline', fontsize=16,
        ha='center', fontweight='bold')

# Input image
draw_box(ax, 0.2, 7.5, 2.2, 1.5, 'Obraz\n224x224x3\n(RGB)', '#BBDEFB', fontsize=11)

# Arrow
draw_arrow(ax, 2.5, 8.25, 3.2, 8.25)
ax.text(2.85, 8.55, 'podziel', fontsize=8, ha='center', color='gray')

# Patches
draw_box(ax, 3.2, 7.5, 2.5, 1.5, '196 patchy\n16x16x3\nkażdy = 768 dim', '#C8E6C9', fontsize=10)

# Arrow
draw_arrow(ax, 5.8, 8.25, 6.5, 8.25)
ax.text(6.15, 8.55, r'$\mathbf{E} \in \mathbb{R}^{768 \times 384}$', fontsize=9, ha='center', color='#B71C1C')

# Linear projection + position embeddings
draw_box(ax, 6.5, 7.5, 3.0, 1.5, 'Projekcja liniowa\n768 → 384 dim\n+ pozycja + [CLS]', '#FFF9C4', fontsize=10)

# Arrow down
draw_arrow(ax, 8.0, 7.4, 8.0, 6.5)
ax.text(8.3, 6.95, '197 tokenów x 384d', fontsize=8, color='gray')

# Transformer Block 1
draw_box(ax, 5.5, 4.8, 5.0, 1.5, 'Blok Transformera x12\n'
         'Layer Norm → Self-Attention → Layer Norm → MLP\n'
         '(szczegóły poniżej)', '#E1BEE7', fontsize=10)

# Arrow down
draw_arrow(ax, 8.0, 4.7, 8.0, 3.8)

# Output
draw_box(ax, 5.5, 2.3, 5.0, 1.3, 'Wyjście: 197 tokenów x 384 dim\n'
         '[CLS] = globalny embedding obrazu', '#FFCCBC', fontsize=10)

# Arrow to CLS
draw_arrow(ax, 8.0, 2.2, 8.0, 1.3)
ax.text(8.3, 1.75, 'weź [CLS]', fontsize=9, color='gray')

# Final embedding
draw_box(ax, 6.0, 0.3, 4.0, 0.8, r'Embedding: $\mathbf{z} \in \mathbb{R}^{384}$',
         '#A5D6A7', fontsize=12, text_color='#1B5E20')

# --- Self-Attention detail (right side) ---
ax.text(13.5, 9.3, 'Self-Attention\n(wewnątrz bloku)', fontsize=12,
        ha='center', fontweight='bold', color='#4A148C')

# Q, K, V
draw_box(ax, 11.5, 7.8, 1.1, 0.7, 'Q\nQuery', '#E3F2FD', fontsize=9)
draw_box(ax, 12.8, 7.8, 1.1, 0.7, 'K\nKey', '#E8F5E9', fontsize=9)
draw_box(ax, 14.1, 7.8, 1.1, 0.7, 'V\nValue', '#FFF3E0', fontsize=9)

ax.text(13.5, 7.4, r'Każdy token $\to$ Q, K, V', fontsize=9, ha='center', color='gray')

# Attention scores
draw_box(ax, 12.0, 6.2, 3.0, 0.9, r'Uwaga = softmax(QK$^T$/√d)', '#F3E5F5', fontsize=9)
draw_arrow(ax, 12.05, 7.7, 12.5, 7.2)
draw_arrow(ax, 13.35, 7.7, 13.5, 7.2)

# Weighted sum
draw_box(ax, 12.0, 5.0, 3.0, 0.9, 'Wynik = Uwaga × V\n(ważona suma wartości)', '#E0F7FA', fontsize=9)
draw_arrow(ax, 13.5, 6.1, 13.5, 6.0)
draw_arrow(ax, 14.65, 7.7, 14.65, 5.95)

# Explanation
ax.text(13.5, 4.3, 'Każdy token "patrzy" na\nwszystkie inne tokeny\ni zbiera informację',
        fontsize=10, ha='center', style='italic', color='#4A148C',
        bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8))

# MLP detail
ax.text(13.5, 3.2, 'MLP (po attention):', fontsize=10,
        ha='center', fontweight='bold', color='#E65100')
draw_box(ax, 11.8, 2.0, 3.4, 0.9,
         '384 → 1536 → 384\n(rozszerz 4x, potem skompresuj)', '#FFF3E0', fontsize=9)

# Random init note
ax.text(13.5, 1.0, 'Przy losowej inicjalizacji:\nwagi Q,K,V,MLP = losowe\n→ embedding = losowy szum\n→ trening uczy sensownych wag',
        fontsize=9, ha='center', color='#B71C1C',
        bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.9))

# Left side annotations
ax.text(1.5, 6.5, 'Embedding pozycyjny:\nuczony wektor dla\nkażdej pozycji 1..196\n→ sieć wie GDZIE\njest dany patch',
        fontsize=9, ha='center', color='#00695C',
        bbox=dict(boxstyle='round', facecolor='#E0F2F1', alpha=0.9))

ax.text(1.5, 4.0, '[CLS] token:\ndodatkowy uczony\nwektor "pytający"\n→ zbiera informację\nz WSZYSTKICH patchy\nprzez self-attention',
        fontsize=9, ha='center', color='#1565C0',
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.9))

ax.text(1.5, 1.5, 'ViT-Small:\n12 bloków\n6 głów attention\n384 dim embedding\n~22M parametrów',
        fontsize=9, ha='center', color='#333',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))

plt.savefig(os.path.join(FIGURES_DIR, 'vit_pipeline.pdf'), bbox_inches='tight', dpi=300)
print("Plot 2 (pipeline) saved.")

# ============================================================
# Plot 3: Self-attention visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: Patches with attention from CLS
ax = axes[0]
grid = np.zeros((14, 14, 3)) + 0.85
# Simulate attention weights from CLS
np.random.seed(12)
attn = np.random.rand(14, 14) ** 3
attn[5:9, 4:10] = 0.8 + np.random.rand(4, 6) * 0.2  # High attention region
attn = attn / attn.max()

for i in range(14):
    for j in range(14):
        a = attn[i, j]
        grid[i, j] = [1, 1-a, 1-a]  # Red = high attention

ax.imshow(grid, interpolation='nearest')
for i in range(15):
    ax.axhline(i - 0.5, color='gray', linewidth=0.3)
    ax.axvline(i - 0.5, color='gray', linewidth=0.3)
ax.set_title('[CLS] attention weights\n(na które patche "patrzy" [CLS])', fontsize=12, fontweight='bold')
ax.set_xlabel('patch kolumna', fontsize=10)
ax.set_ylabel('patch wiersz', fontsize=10)
ax.text(7, -1.5, r'Czerwony = wysoka uwaga $\alpha_{ij}$', fontsize=10,
        ha='center', color='#B71C1C')

# Panel 2: Attention matrix (token x token)
ax = axes[1]
N_tok = 10  # show 10 tokens for clarity
attn_matrix = np.random.rand(N_tok, N_tok) ** 2
attn_matrix = attn_matrix / attn_matrix.sum(axis=1, keepdims=True)  # softmax-like
im = ax.imshow(attn_matrix, cmap='Reds', aspect='equal')
plt.colorbar(im, ax=ax, fraction=0.046)
labels = ['[CLS]'] + [f'P{i}' for i in range(1, N_tok)]
ax.set_xticks(range(N_tok))
ax.set_xticklabels(labels, fontsize=8, rotation=45)
ax.set_yticks(range(N_tok))
ax.set_yticklabels(labels, fontsize=8)
ax.set_title('Macierz attention\n(kto na kogo patrzy)', fontsize=12, fontweight='bold')
ax.set_xlabel('Key (źródło informacji)', fontsize=10)
ax.set_ylabel('Query (kto pyta)', fontsize=10)

# Panel 3: Training from scratch
ax = axes[2]
epochs = np.arange(0, 101)
# Simulated curves
loss = 3.0 * np.exp(-epochs / 25) + 0.3 + np.random.randn(101) * 0.05
attn_entropy = 6.0 * np.exp(-epochs / 30) + 1.0  # High entropy = uniform attention
embed_quality = 1 - np.exp(-epochs / 20)

ax2 = ax.twinx()
ax.plot(epochs, loss, 'b-', linewidth=2, label='Loss (maleje)', alpha=0.8)
ax.plot(epochs, attn_entropy, 'r--', linewidth=2, label='Entropia attention (maleje)', alpha=0.8)
ax2.plot(epochs, embed_quality * 100, 'g-', linewidth=2, label='Jakość embeddingów (%)', alpha=0.8)

ax.set_xlabel('Epoka', fontsize=11)
ax.set_ylabel('Loss / Entropia', fontsize=11, color='blue')
ax2.set_ylabel(r'Jakość embeddingów (%)', fontsize=11, color='green')
ax.set_title('Trening od zera:\nod losowego szumu do sensownych cech', fontsize=12, fontweight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='center right')

# Annotations
ax.annotate('Epoka 0:\nwagi losowe\n= embedding = szum',
            xy=(0, 3.0), xytext=(15, 4.5), fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'),
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.annotate('Epoka ~50:\nattention uczy się\nna co patrzeć',
            xy=(50, 1.3), xytext=(55, 3.0), fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'vit_attention.pdf'), bbox_inches='tight', dpi=300)
print("Plot 3 (attention) saved.")

# ============================================================
# Plot 4: Position embeddings visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: position embedding grid
ax = axes[0]
np.random.seed(42)
pos_emb = np.random.randn(14, 14)
# Make it structured (nearby positions similar)
from scipy.ndimage import gaussian_filter
pos_emb = gaussian_filter(pos_emb, sigma=2)

im = ax.imshow(pos_emb, cmap='coolwarm', interpolation='nearest')
plt.colorbar(im, ax=ax, fraction=0.046)
for i in range(15):
    ax.axhline(i - 0.5, color='gray', linewidth=0.3)
    ax.axvline(i - 0.5, color='gray', linewidth=0.3)
ax.set_title('Embedding pozycyjny\n(1 wymiar z 384)', fontsize=12, fontweight='bold')
ax.set_xlabel('kolumna patcha', fontsize=10)
ax.set_ylabel('wiersz patcha', fontsize=10)
ax.text(7, -1.8, 'Blisko = podobne wartości\n→ sieć wie, które patche sąsiadują',
        fontsize=10, ha='center', color='#1565C0')

# Right: cosine similarity of position embeddings
ax = axes[1]
# Simulate cosine similarity matrix for position embeddings
N_pos = 14 * 14
pos_vecs = np.random.randn(N_pos, 384)
# Make nearby positions similar
for idx in range(N_pos):
    row, col = idx // 14, idx % 14
    for idx2 in range(N_pos):
        r2, c2 = idx2 // 14, idx2 % 14
        dist = np.sqrt((row - r2)**2 + (col - c2)**2)
        if dist < 4:
            pos_vecs[idx2] += pos_vecs[idx] * np.exp(-dist / 2) * 0.5

# Compute similarity for selected positions
selected = [0, 7, 13, 91, 97, 103, 182, 189, 195]  # 3x3 grid from corners and center
labels_pos = ['(0,0)', '(0,7)', '(0,13)', '(6,7)', '(7,0)', '(7,6)', '(13,0)', '(13,7)', '(13,13)']
sim_matrix = np.zeros((len(selected), len(selected)))
for i, si in enumerate(selected):
    for j, sj in enumerate(selected):
        cos_sim = np.dot(pos_vecs[si], pos_vecs[sj]) / (np.linalg.norm(pos_vecs[si]) * np.linalg.norm(pos_vecs[sj]))
        sim_matrix[i, j] = cos_sim

im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
plt.colorbar(im, ax=ax, fraction=0.046)
ax.set_xticks(range(len(selected)))
ax.set_xticklabels(labels_pos, fontsize=8, rotation=45)
ax.set_yticks(range(len(selected)))
ax.set_yticklabels(labels_pos, fontsize=8)
ax.set_title('Podobieństwo embeddingów pozycji\n(cosine similarity)', fontsize=12, fontweight='bold')
ax.text(4, -1.8, 'Bliskie pozycje = wysokie podobieństwo (zielony)\nDalekie = niskie (czerwony)',
        fontsize=10, ha='center', color='#1565C0')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'vit_positions.pdf'), bbox_inches='tight', dpi=300)
print("Plot 4 (positions) saved.")

print("\nAll ViT plots generated!")
