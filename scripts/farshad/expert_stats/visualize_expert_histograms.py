#!/usr/bin/env python3
"""
Visualize expert activation histograms across all layers to show similarity/differences.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

# Set style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Read the CSV (look for it in statistics/ subdirectory)
import sys
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = "bench-throughput-logs-qwen3-exps-bf16-20260508/statistics/expert_activation_histogram.csv"
df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} rows from {csv_path}")
print(f"Layers: {df['Layer'].min()} to {df['Layer'].max()}")
print(f"Activation counts range: {df['Activations_Per_Expert'].min()} to {df['Activations_Per_Expert'].max()}")

# Create output directory (in statistics/ subdirectory next to CSV)
csv_dir = Path(csv_path).parent
output_dir = csv_dir
print(f"\nOutput directory: {output_dir}")

# ============================================================================
# 1. HEATMAP: Num_Experts for each (Layer, Activations_Per_Expert)
# ============================================================================
print("\n" + "="*80)
print("Creating heatmap: Expert count by layer and activation frequency")
print("="*80)

# Pivot table: rows = activations/expert, cols = layers, values = num experts
pivot = df.pivot_table(
    index='Activations_Per_Expert',
    columns='Layer',
    values='Num_Experts',
    fill_value=0
)

fig, ax = plt.subplots(figsize=(20, 8))
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Number of Experts', fontsize=11)

# Set ticks and labels
ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_xticklabels(pivot.columns)
ax.set_yticklabels(pivot.index)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Activations per Expert', fontsize=12)
ax.set_title('Expert Activation Distribution Across Layers\n(Heatmap shows how many experts are activated N times)', 
             fontsize=14, fontweight='bold')

heatmap_path = output_dir / "expert_heatmap.png"
plt.tight_layout()
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {heatmap_path}")
plt.close()

# ============================================================================
# 2. LINE PLOT: Distribution of activation counts across layers
# ============================================================================
print("\n" + "="*80)
print("Creating line plot: Expert distribution patterns")
print("="*80)

fig, ax = plt.subplots(figsize=(16, 8))

# Plot each layer's histogram as a line
for layer in sorted(df['Layer'].unique()):
    layer_data = df[df['Layer'] == layer].sort_values('Activations_Per_Expert')
    
    # Only show every 6th layer for clarity (8 lines total)
    if layer % 6 == 0 or layer == 47:
        ax.plot(
            layer_data['Activations_Per_Expert'],
            layer_data['Num_Experts'],
            marker='o',
            linewidth=2,
            markersize=6,
            label=f'Layer {layer}',
            alpha=0.7
        )

ax.set_xlabel('Activations per Expert', fontsize=12)
ax.set_ylabel('Number of Experts', fontsize=12)
ax.set_title('Expert Activation Distributions (Selected Layers)\nShows how many experts are activated N times per decode step',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

lineplot_path = output_dir / "expert_distributions.png"
plt.tight_layout()
plt.savefig(lineplot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {lineplot_path}")
plt.close()

# ============================================================================
# 3. STATISTICS: Layer similarity metrics
# ============================================================================
print("\n" + "="*80)
print("Computing layer similarity statistics")
print("="*80)

# For each layer, compute statistics
layer_stats = []
for layer in sorted(df['Layer'].unique()):
    layer_data = df[df['Layer'] == layer]
    
    # Weighted mean activations per expert
    total_activations = (layer_data['Activations_Per_Expert'] * layer_data['Num_Experts']).sum()
    total_experts = layer_data['Num_Experts'].sum()
    mean_activations = total_activations / total_experts
    
    # Weighted std
    variance = ((layer_data['Activations_Per_Expert'] - mean_activations)**2 * layer_data['Num_Experts']).sum() / total_experts
    std_activations = np.sqrt(variance)
    
    # Max activation count
    max_activations = layer_data['Activations_Per_Expert'].max()
    
    # Number of unique activation buckets
    num_buckets = len(layer_data)
    
    layer_stats.append({
        'Layer': layer,
        'Total_Experts': total_experts,
        'Mean_Activations_Per_Expert': mean_activations,
        'Std_Activations': std_activations,
        'Max_Activations': max_activations,
        'Num_Buckets': num_buckets
    })

stats_df = pd.DataFrame(layer_stats)

print("\nLayer Statistics Summary:")
print(f"  Mean activations/expert:  {stats_df['Mean_Activations_Per_Expert'].mean():.2f} ± {stats_df['Mean_Activations_Per_Expert'].std():.2f}")
print(f"  Std across layers:        {stats_df['Std_Activations'].mean():.2f} ± {stats_df['Std_Activations'].std():.2f}")
print(f"  Total experts per layer:  {stats_df['Total_Experts'].mean():.1f} ± {stats_df['Total_Experts'].std():.1f}")
print(f"  Num buckets per layer:    {stats_df['Num_Buckets'].mean():.1f} ± {stats_df['Num_Buckets'].std():.1f}")

# ============================================================================
# 4. BOX PLOT: Distribution of expert counts across activation buckets
# ============================================================================
print("\n" + "="*80)
print("Creating box plot: Expert count distribution by activation frequency")
print("="*80)

# Group by activation count to see distribution across layers
fig, ax = plt.subplots(figsize=(14, 7))

activation_counts = sorted(df['Activations_Per_Expert'].unique())
data_for_boxplot = []
labels = []

for act_count in activation_counts:
    data = df[df['Activations_Per_Expert'] == act_count]['Num_Experts'].values
    if len(data) > 0:
        data_for_boxplot.append(data)
        labels.append(f'{act_count}')

bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                showmeans=True, meanline=True)

# Color the boxes
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax.set_xlabel('Activations per Expert', fontsize=12)
ax.set_ylabel('Number of Experts (distribution across layers)', fontsize=12)
ax.set_title('Distribution of Expert Counts Across Layers\n(Box plot shows variation across 48 layers for each activation frequency)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

boxplot_path = output_dir / "expert_boxplot.png"
plt.tight_layout()
plt.savefig(boxplot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {boxplot_path}")
plt.close()

# ============================================================================
# 5. STACKED AREA: Composition of each layer
# ============================================================================
print("\n" + "="*80)
print("Creating stacked area chart: Layer composition")
print("="*80)

# Create a matrix where rows = layers, columns = activation counts
max_act = df['Activations_Per_Expert'].max()
composition_matrix = np.zeros((48, max_act))

for _, row in df.iterrows():
    layer = int(row['Layer'])
    act = int(row['Activations_Per_Expert'])
    num_exp = int(row['Num_Experts'])
    composition_matrix[layer, act-1] = num_exp

fig, ax = plt.subplots(figsize=(16, 8))

# Stack area plot
layers = np.arange(48)
ax.stackplot(
    layers,
    *[composition_matrix[:, i] for i in range(max_act)],
    labels=[f'{i+1} act/expert' for i in range(max_act)],
    alpha=0.8
)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Number of Experts', fontsize=12)
ax.set_title('Layer Composition: Expert Distribution by Activation Frequency\n(Stacked area shows contribution of each activation count)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

stackplot_path = output_dir / "expert_stackplot.png"
plt.tight_layout()
plt.savefig(stackplot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {stackplot_path}")
plt.close()

# ============================================================================
# 6. LAYER-TO-LAYER SIMILARITY MATRIX
# ============================================================================
print("\n" + "="*80)
print("Computing layer-to-layer similarity matrix")
print("="*80)

# Create feature vectors for each layer (using num_experts for each activation count)
layer_vectors = np.zeros((48, max_act))
for _, row in df.iterrows():
    layer = int(row['Layer'])
    act = int(row['Activations_Per_Expert'])
    num_exp = int(row['Num_Experts'])
    layer_vectors[layer, act-1] = num_exp

# Compute cosine similarity between all layer pairs
similarity_matrix = np.zeros((48, 48))
for i in range(48):
    for j in range(48):
        similarity_matrix[i, j] = cosine_similarity(layer_vectors[i], layer_vectors[j])

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.9, vmax=1.0, aspect='auto')

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Layer', fontsize=12)
ax.set_title('Layer-to-Layer Similarity (Cosine Similarity)\nDarker green = more similar patterns',
             fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Cosine Similarity', fontsize=11)

# Set ticks
ax.set_xticks(np.arange(0, 48, 4))
ax.set_yticks(np.arange(0, 48, 4))

similarity_path = output_dir / "layer_similarity.png"
plt.tight_layout()
plt.savefig(similarity_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {similarity_path}")
plt.close()

# Print similarity statistics
print("\nSimilarity Statistics:")
# Exclude diagonal (self-similarity = 1.0)
off_diagonal = similarity_matrix[~np.eye(48, dtype=bool)]
print(f"  Mean similarity (excluding self): {off_diagonal.mean():.4f}")
print(f"  Min similarity:  {off_diagonal.min():.4f}")
print(f"  Max similarity:  {off_diagonal.max():.4f}")
print(f"  Std similarity:  {off_diagonal.std():.4f}")

# Find most similar and most different layer pairs
min_idx = np.unravel_index(np.argmin(similarity_matrix + np.eye(48) * 10), similarity_matrix.shape)
max_idx = np.unravel_index(np.argmax(similarity_matrix - np.eye(48)), similarity_matrix.shape)
print(f"\n  Most different layers: {min_idx[0]} and {min_idx[1]} (similarity: {similarity_matrix[min_idx]:.4f})")
print(f"  Most similar layers:   {max_idx[0]} and {max_idx[1]} (similarity: {similarity_matrix[max_idx]:.4f})")

print("\n" + "="*80)
print("Visualization complete!")
print("="*80)
print(f"\nAll visualizations saved to: {output_dir}/")
print(f"  1. expert_heatmap.png - Heatmap of expert counts")
print(f"  2. expert_distributions.png - Line plot of selected layers")
print(f"  3. expert_boxplot.png - Box plot showing variation")
print(f"  4. expert_stackplot.png - Stacked area chart")
print(f"  5. layer_similarity.png - Layer-to-layer similarity matrix")
