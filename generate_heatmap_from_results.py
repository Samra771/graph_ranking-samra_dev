"""
generate_heatmap_from_results.py
=================================
Run this script AFTER experiment2_generalisation.py has completed.

It takes the results you already computed and:
1. Saves them to CSV correctly (UTF-8 encoding — fixes Windows error)
2. Generates the generalisation heatmap figures for betweenness and closeness centrality.
No need to re-run the expensive training — just paste your results below.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "./paper_figures_generalisation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# PASTE YOUR RESULTS HERE
# These are taken directly from your terminal output
# ═════════════════════════════════════════════════════════════════════════════

results = {
    # Betweenness — ER trained model
    "ER->ER (Betweenness)"    : (0.8544, 0.0104),
    "ER->BA (Betweenness)"    : (0.8399, 0.0119),
    "ER->GRP (Betweenness)"   : (0.7293, 0.0485),

    # Closeness — ER trained model
    "ER->ER (Closeness)"      : (0.8260, 0.0112),
    "ER->BA (Closeness)"      : (0.6037, 0.0273),
    "ER->GRP (Closeness)"     : (0.0171, 0.0494),

    # Betweenness — Mixed trained model
    "Mixed->ER (Betweenness)" : (0.8867, 0.0095),
    "Mixed->BA (Betweenness)" : (0.9334, 0.0048),
    "Mixed->GRP (Betweenness)": (0.8781, 0.0103),
}

# ═════════════════════════════════════════════════════════════════════════════
# SAVE CSV WITH UTF-8 ENCODING (fixes Windows cp1252 error)
# ═════════════════════════════════════════════════════════════════════════════
csv_path = f"{OUTPUT_DIR}/generalisation_results.csv"
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("Condition,Mean KT,Std KT\n")
    for key, (mean, std) in results.items():
        f.write(f"{key},{mean:.6f},{std:.6f}\n")
print(f"Saved: {csv_path}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE STYLE
# ═════════════════════════════════════════════════════════════════════════════
STYLE = {
    "figure.dpi"     : 300,
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 12,
    "axes.labelsize" : 11,
}
plt.rcParams.update(STYLE)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Betweenness Heatmap (ER-trained vs Mixed-trained)
# ═════════════════════════════════════════════════════════════════════════════
train_types = ["ER-trained", "Mixed-trained"]
test_types  = ["ER", "BA", "GRP"]

bet_matrix = np.array([
    [results["ER->ER (Betweenness)"][0],
     results["ER->BA (Betweenness)"][0],
     results["ER->GRP (Betweenness)"][0]],
    [results["Mixed->ER (Betweenness)"][0],
     results["Mixed->BA (Betweenness)"][0],
     results["Mixed->GRP (Betweenness)"][0]],
])

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(bet_matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Kendall tau", fontsize=11)

ax.set_xticks(range(len(test_types)))
ax.set_xticklabels(test_types, fontsize=11)
ax.set_yticks(range(len(train_types)))
ax.set_yticklabels(train_types, fontsize=11)
ax.set_xlabel("Test Graph Type")
ax.set_ylabel("Training Condition")
ax.set_title("Betweenness Centrality: Generalisation Across Graph Types")

# Annotate each cell with the tau value
for i in range(len(train_types)):
    for j in range(len(test_types)):
        val     = bet_matrix[i, j]
        color   = "white" if val < 0.5 else "black"
        std_val = results[
            f"{'ER' if i==0 else 'Mixed'}->{test_types[j]} (Betweenness)"
        ][1]
        ax.text(j, i, f"tau = {val:.3f}\n+/- {std_val:.3f}",
                ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

fig.tight_layout()
path = f"{OUTPUT_DIR}/fig_generalisation_heatmap_betweenness.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Closeness Heatmap (ER-trained only, three test types)
# ═════════════════════════════════════════════════════════════════════════════
close_vals = np.array([[
    results["ER->ER (Closeness)"][0],
    results["ER->BA (Closeness)"][0],
    results["ER->GRP (Closeness)"][0],
]])

fig, ax = plt.subplots(figsize=(8, 2.5))
im = ax.imshow(close_vals, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Kendall tau", fontsize=11)

ax.set_xticks(range(len(test_types)))
ax.set_xticklabels(test_types, fontsize=11)
ax.set_yticks([0])
ax.set_yticklabels(["ER-trained"], fontsize=11)
ax.set_xlabel("Test Graph Type")
ax.set_ylabel("Training Condition")
ax.set_title("Closeness Centrality: Generalisation Across Graph Types")

close_keys = ["ER->ER (Closeness)", "ER->BA (Closeness)", "ER->GRP (Closeness)"]
for j, key in enumerate(close_keys):
    val   = results[key][0]
    std_v = results[key][1]
    color = "white" if val < 0.4 else "black"
    ax.text(j, 0, f"tau = {val:.3f}\n+/- {std_v:.3f}",
            ha="center", va="center",
            fontsize=10, color=color, fontweight="bold")

fig.tight_layout()
path = f"{OUTPUT_DIR}/fig_generalisation_heatmap_closeness.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Side by side bar chart comparing ER-trained vs Mixed-trained
# ═════════════════════════════════════════════════════════════════════════════
x       = np.arange(len(test_types))
width   = 0.35

er_means  = [results[f"ER->{t} (Betweenness)"][0]    for t in test_types]
er_stds   = [results[f"ER->{t} (Betweenness)"][1]    for t in test_types]
mix_means = [results[f"Mixed->{t} (Betweenness)"][0]  for t in test_types]
mix_stds  = [results[f"Mixed->{t} (Betweenness)"][1]  for t in test_types]

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, er_means,  width, yerr=er_stds,
               label="ER-trained",    color="#6baed6",
               capsize=5, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, mix_means, width, yerr=mix_stds,
               label="Mixed-trained", color="#2ca02c",
               capsize=5, edgecolor="black", linewidth=0.5)

ax.set_ylabel("Kendall tau")
ax.set_title("Betweenness Centrality: ER-trained vs Mixed-trained Model")
ax.set_xticks(x)
ax.set_xticklabels(["ER graphs", "BA graphs", "GRP graphs"])
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars1, er_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars2, mix_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
path = f"{OUTPUT_DIR}/fig_generalisation_bar_chart.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ═════════════════════════════════════════════════════════════════════════════
# PRINT TABLE
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  RESULTS TABLE")
print(f"{'='*65}")
print(f"\n  Table: Betweenness Centrality Generalization\n")
print(f"  {'Training':15} {'Test: ER':>15} {'Test: BA':>15} {'Test: GRP':>15}")
print(f"  {'-'*60}")
for train in ["ER", "Mixed"]:
    label = f"ER-trained" if train == "ER" else "Mixed-trained"
    vals  = [results[f"{train}->{t} (Betweenness)"] for t in test_types]
    row   = f"  {label:15}"
    for m, s in vals:
        row += f"  {m:.3f}+/-{s:.3f}"
    print(row)

print(f"\n  Table: Closeness Centrality Generalization\n")
print(f"  {'Training':15} {'Test: ER':>15} {'Test: BA':>15} {'Test: GRP':>15}")
print(f"  {'-'*60}")
vals  = [results[f"ER->{t} (Closeness)"] for t in test_types]
row   = f"  {'ER-trained':15}"
for m, s in vals:
    row += f"  {m:.3f}+/-{s:.3f}"
print(row)

# ═════════════════════════════════════════════════════════════════════════════
# PRINT KEY FINDINGS 
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  KEY FINDINGS")
print(f"{'='*65}")

bet_er_ba_drop  = results["ER->ER (Betweenness)"][0] - results["ER->BA (Betweenness)"][0]
bet_er_grp_drop = results["ER->ER (Betweenness)"][0] - results["ER->GRP (Betweenness)"][0]
mix_improvement_ba  = results["Mixed->BA (Betweenness)"][0] - results["ER->BA (Betweenness)"][0]
mix_improvement_grp = results["Mixed->GRP (Betweenness)"][0] - results["ER->GRP (Betweenness)"][0]

print(f"""
  BETWEENNESS:
  - ER-trained model drops only {bet_er_ba_drop:.3f} tau when tested on BA graphs
    (ER: {results['ER->ER (Betweenness)'][0]:.3f} -> BA: {results['ER->BA (Betweenness)'][0]:.3f})
    This is strong out-of-distribution generalization.

  - ER-trained model drops {bet_er_grp_drop:.3f} tau on GRP (community structure graphs)
    (ER: {results['ER->ER (Betweenness)'][0]:.3f} -> GRP: {results['ER->GRP (Betweenness)'][0]:.3f})
    GRP graphs have fundamentally different topology — some drop is expected.

  - Mixed training IMPROVES over ER-only on all graph types:
    BA improvement  : +{mix_improvement_ba:.3f} tau ({results['ER->BA (Betweenness)'][0]:.3f} -> {results['Mixed->BA (Betweenness)'][0]:.3f})
    GRP improvement : +{mix_improvement_grp:.3f} tau ({results['ER->GRP (Betweenness)'][0]:.3f} -> {results['Mixed->GRP (Betweenness)'][0]:.3f})
    ER improvement  : +{results['Mixed->ER (Betweenness)'][0]-results['ER->ER (Betweenness)'][0]:.3f} tau
    Mixed training achieves tau > 0.87 on ALL three graph types simultaneously.

  CLOSENESS:
  - Strong on ER ({results['ER->ER (Closeness)'][0]:.3f}) and reasonable on BA ({results['ER->BA (Closeness)'][0]:.3f})
  - Fails completely on GRP ({results['ER->GRP (Closeness)'][0]:.3f})
    This is an important limitation to state honestly in the paper.
    GRP graphs have tight community clusters where closeness depends heavily
    on community membership — a very different structure from random ER graphs.
  - RECOMMENDATION: Train a mixed closeness model to fix GRP performance.
    This is easy to add — same approach as Condition D for betweenness.
""")

print(f"  All figures saved to: {OUTPUT_DIR}/")
print(f"  Use these files in your paper:")
print(f"    fig_generalization_heatmap_betweenness.png  -> Figure 4.8")
print(f"    fig_generalization_heatmap_closeness.png    -> Figure 4.9")
print(f"    fig_generalization_bar_chart.png            -> Figure 4.10")
