"""
run_experiment.py
=================
Complete experiment script for GNN Betweenness Centrality Approximation.

This script:
1. Generates 2000 Erdos-Renyi graphs with 200 nodes
2. Trains the GNN model for betweenness centrality
3. Saves training curves (loss, Kendall tau)
4. Generates comparison plots (GNN predictions vs NetworkX ground truth)

Run from your repo folder:
    python run_experiment.py

All figures are saved in ./paper_figures/ at 300 DPI.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend — works on any machine
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.sparse import csr_matrix
import time
import random

# ── Import your existing repo files ──────────────────────────────────────────
# Make sure this script is in the same folder as your repo files
from graph_mgmt_library import Graph
from betweennes_model import GNN_Bet
from betweennes_training_library import sparse_mx_to_torch_sparse_tensor, loss_cal, ranking_correlation

# ── Set random seeds for reproducibility ─────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT SETTINGS — these match your paper exactly
# ═════════════════════════════════════════════════════════════════════════════
GRAPH_NODES      = 200       # Number of nodes per graph
GRAPH_SPARSENESS = 0.15      # Edge probability for Erdos-Renyi (p = 0.15)
TOTAL_GRAPHS     = 2000      # Total graphs generated (1600 train, 400 test)
TRAIN_RATIO      = 0.80      # 80% training, 20% test

HIDDEN_LAYERS    = 20        # GNN hidden layer size
EPOCHS           = 100       # Training epochs
LEARNING_RATE    = 1e-4      # Adam learning rate
DROPOUT          = 0.2       # Dropout probability
WEIGHT_DECAY     = 0.01      # L2 regularisation coefficient
BATCH_SIZE       = 16        # Graphs per batch update

MODEL_SIZE       = GRAPH_NODES  # Must match node count

OUTPUT_DIR       = "./paper_figures"   # Where figures are saved
MODEL_SAVE_PATH  = "./betweenness_model.pth"

# ═════════════════════════════════════════════════════════════════════════════
# SETUP
# ═════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  GNN Betweenness Centrality Experiment")
print(f"{'='*60}")
print(f"  Device     : {device}")
print(f"  Nodes      : {GRAPH_NODES}")
print(f"  Graphs     : {TOTAL_GRAPHS} ({int(TOTAL_GRAPHS*TRAIN_RATIO)} train / {int(TOTAL_GRAPHS*(1-TRAIN_RATIO))} test)")
print(f"  Epochs     : {EPOCHS}")
print(f"  Output dir : {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE GRAPHS
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 1: Generating graphs...")
G = Graph(GRAPH_NODES, GRAPH_SPARSENESS)

adj_matrices        = []
adj_matrices_T      = []
betweenness_list    = []
node_counts         = []

for i in range(TOTAL_GRAPHS):
    # graph_type=2 means Erdos-Renyi (directed)
    graph = G.create_graph(graph_type=2)

    adj  = G.get_full_adjacency_matrix(graph)
    adj_T = adj.transpose()
    bc   = G.get_betweenness_centrality(graph)
    n    = len(G.get_graph_nodes(graph))

    adj_matrices.append(adj)
    adj_matrices_T.append(adj_T)
    betweenness_list.append(bc)
    node_counts.append(n)

    if (i + 1) % 200 == 0:
        print(f"  Generated {i+1}/{TOTAL_GRAPHS} graphs")

# ─ Train/test split ───────────────────────────────────────────────────────────
indices       = list(range(TOTAL_GRAPHS))
random.shuffle(indices)
train_size    = int(TOTAL_GRAPHS * TRAIN_RATIO)
train_indices = set(indices[:train_size])
test_indices  = set(indices[train_size:])

adj_train    = [adj_matrices[i]   for i in range(TOTAL_GRAPHS) if i in train_indices]
adj_T_train  = [adj_matrices_T[i] for i in range(TOTAL_GRAPHS) if i in train_indices]
bc_train     = [betweenness_list[i] for i in range(TOTAL_GRAPHS) if i in train_indices]
nodes_train  = [node_counts[i]    for i in range(TOTAL_GRAPHS) if i in train_indices]

adj_test     = [adj_matrices[i]   for i in range(TOTAL_GRAPHS) if i in test_indices]
adj_T_test   = [adj_matrices_T[i] for i in range(TOTAL_GRAPHS) if i in test_indices]
bc_test      = [betweenness_list[i] for i in range(TOTAL_GRAPHS) if i in test_indices]
nodes_test   = [node_counts[i]    for i in range(TOTAL_GRAPHS) if i in test_indices]

print(f"  Train: {len(adj_train)} graphs | Test: {len(adj_test)} graphs\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD MODEL
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 2: Building model...")
network   = GNN_Bet(
    ninput=MODEL_SIZE,
    nhid=HIDDEN_LAYERS,
    dropout=DROPOUT,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
model, _  = network.model_to_device(network)
model     = model.to(device)
optimizer = network.get_optimizer(model)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def bc_dict_to_array(bc_dict, num_nodes):
    """
    Convert a NetworkX betweenness centrality dictionary
    {node_id: value} to a numpy array of length num_nodes.
    Fills missing nodes with 0.
    """
    arr = np.zeros(num_nodes, dtype=np.float32)
    for node, val in bc_dict.items():
        if node < num_nodes:
            arr[node] = val
    return arr


def train_one_epoch(adj_list, adj_t_list, bc_dicts, node_list):
    """
    Runs one full training epoch over all training graphs.
    Returns average training loss for this epoch.
    """
    model.train()
    total_loss    = 0.0
    num_samples   = len(adj_list)
    perm          = np.random.permutation(num_samples)

    for batch_start in range(0, num_samples, BATCH_SIZE):
        batch_idx = perm[batch_start : batch_start + BATCH_SIZE]
        optimizer.zero_grad()
        batch_loss = 0.0

        for i in batch_idx:
            adj   = sparse_mx_to_torch_sparse_tensor(adj_list[i],   device)
            adj_t = sparse_mx_to_torch_sparse_tensor(adj_t_list[i], device)
            n     = node_list[i]

            y_out    = model(adj, adj_t)
            true_arr = bc_dict_to_array(bc_dicts[i], MODEL_SIZE)
            true_val = torch.from_numpy(true_arr).float().to(device)

            loss = loss_cal(y_out, true_val, n, device, MODEL_SIZE)
            batch_loss += loss

        batch_loss = batch_loss / len(batch_idx)
        batch_loss.backward()
        optimizer.step()
        total_loss += float(batch_loss)

    return total_loss / max(1, num_samples // BATCH_SIZE)


def evaluate(adj_list, adj_t_list, bc_dicts, node_list):
    """
    Evaluates model on a set of graphs.
    Returns (average_loss, average_kendall_tau, std_kendall_tau).
    """
    model.eval()
    total_loss = 0.0
    kt_scores  = []

    with torch.no_grad():
        for i in range(len(adj_list)):
            adj   = sparse_mx_to_torch_sparse_tensor(adj_list[i],   device)
            adj_t = sparse_mx_to_torch_sparse_tensor(adj_t_list[i], device)
            n     = node_list[i]

            y_out    = model(adj, adj_t)
            true_arr = bc_dict_to_array(bc_dicts[i], MODEL_SIZE)
            true_val = torch.from_numpy(true_arr).float().to(device)

            loss = loss_cal(y_out, true_val, n, device, MODEL_SIZE)
            total_loss += float(loss)

            kt = ranking_correlation(y_out, true_val, n, MODEL_SIZE)
            kt_scores.append(kt)

    avg_loss = total_loss / len(adj_list)
    avg_kt   = float(np.mean(kt_scores))
    std_kt   = float(np.std(kt_scores))
    return avg_loss, avg_kt, std_kt


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 3: Training...")

train_losses   = []
test_losses    = []
rel_diffs      = []
kt_means       = []
kt_stds        = []

t0 = time.time()

for epoch in range(EPOCHS):
    tr_loss              = train_one_epoch(adj_train, adj_T_train, bc_train, nodes_train)
    te_loss, kt_m, kt_s = evaluate(adj_test, adj_T_test, bc_test, nodes_test)
    rel_diff             = (tr_loss - te_loss) / max(tr_loss, 1e-9)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    rel_diffs.append(rel_diff)
    kt_means.append(kt_m)
    kt_stds.append(kt_s)

    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Test Loss: {te_loss:.4f} | "
              f"Kendall tau: {kt_m:.4f} ± {kt_s:.4f} | "
              f"Time: {elapsed:.0f}s")

print(f"\n  Training complete. Total time: {time.time()-t0:.0f}s")

# Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"  Model saved to {MODEL_SAVE_PATH}\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE FIGURES
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 4: Generating figures...")

STYLE = {
    "figure.dpi"        : 300,
    "font.family"       : "serif",
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 10,
    "lines.linewidth"   : 1.5,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
}
plt.rcParams.update(STYLE)

epochs_range = range(1, EPOCHS + 1)

# ── Figure 1: Training Loss ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs_range, train_losses, color="#1f77b4", label="Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss vs Epoch (Betweenness Centrality)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_training_loss.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_training_loss.png")

# ── Figure 2: Test Loss ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs_range, test_losses, color="#ff7f0e", label="Test Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Test Loss vs Epoch (Betweenness Centrality)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_test_loss.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_test_loss.png")

# ── Figure 3: Relative Loss Difference ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs_range, rel_diffs, color="#2ca02c", label="Relative Difference")
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("(Train Loss − Test Loss) / Train Loss")
ax.set_title("Relative Loss Difference vs Epoch")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_relative_loss_diff.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_relative_loss_diff.png")

# ── Figure 4: Kendall Tau ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(epochs_range, kt_means, color="#9467bd", label="Kendall τ (mean)")
ax.fill_between(
    epochs_range,
    [m - s for m, s in zip(kt_means, kt_stds)],
    [m + s for m, s in zip(kt_means, kt_stds)],
    alpha=0.2, color="#9467bd", label="± 1 std"
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Kendall τ")
ax.set_title("Kendall Rank Correlation vs Epoch")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_kendall_tau.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_kendall_tau.png")

# ── Figure 5: Comparison Plot — GNN vs NetworkX ───────────────────────────────
# Pick a representative test graph: the one with median Kendall tau
model.eval()
kt_final = []
predictions_all = []
ground_truths_all = []

with torch.no_grad():
    for i in range(len(adj_test)):
        adj   = sparse_mx_to_torch_sparse_tensor(adj_test[i],   device)
        adj_t = sparse_mx_to_torch_sparse_tensor(adj_T_test[i], device)
        n     = nodes_test[i]

        y_out    = model(adj, adj_t)
        true_arr = bc_dict_to_array(bc_test[i], MODEL_SIZE)
        true_val = torch.from_numpy(true_arr).float().to(device)

        kt = ranking_correlation(y_out, true_val, n, MODEL_SIZE)
        kt_final.append(kt)

        pred_np   = y_out.cpu().numpy().flatten()[:n]
        truth_np  = true_arr[:n]
        predictions_all.append(pred_np)
        ground_truths_all.append(truth_np)

# Choose the test graph closest to median Kendall tau
kt_array       = np.array(kt_final)
median_kt      = np.median(kt_array)
best_idx       = int(np.argmin(np.abs(kt_array - median_kt)))
pred_plot      = predictions_all[best_idx]
truth_plot     = ground_truths_all[best_idx]
n_plot         = nodes_test[best_idx]

# Normalise GNN predictions to [0,1] range for fair visual comparison
pred_min, pred_max = pred_plot.min(), pred_plot.max()
if pred_max - pred_min > 1e-9:
    pred_plot_norm = (pred_plot - pred_min) / (pred_max - pred_min)
else:
    pred_plot_norm = pred_plot

truth_min, truth_max = truth_plot.min(), truth_plot.max()
if truth_max - truth_min > 1e-9:
    truth_plot_norm = (truth_plot - truth_min) / (truth_max - truth_min)
else:
    truth_plot_norm = truth_plot

node_indices = np.arange(n_plot)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(node_indices, truth_plot_norm,
        "r--o", markersize=4, linewidth=1.2, label="NetworkX (Ground Truth)")
ax.plot(node_indices, pred_plot_norm,
        "b--o", markersize=4, linewidth=1.2, label="GNN Prediction")
ax.set_xlabel("Node Index")
ax.set_ylabel("Betweenness Centrality (normalised)")
ax.set_title(f"Betweenness Centrality: GNN vs Ground Truth  (Kendall τ = {kt_array[best_idx]:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_betweenness_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_betweenness_comparison.png  (graph #{best_idx}, τ = {kt_array[best_idx]:.3f})")

# ── Figure 6: Combined 2x2 grid (for paper) ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(epochs_range, train_losses, color="#1f77b4")
axes[0, 0].set_title("(a) Training Loss vs Epoch")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")

axes[0, 1].plot(epochs_range, test_losses, color="#ff7f0e")
axes[0, 1].set_title("(b) Test Loss vs Epoch")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")

axes[1, 0].plot(epochs_range, rel_diffs, color="#2ca02c")
axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[1, 0].set_title("(c) Relative Loss Difference")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("(Train − Test) / Train")

axes[1, 1].plot(epochs_range, kt_means, color="#9467bd")
axes[1, 1].fill_between(
    epochs_range,
    [m - s for m, s in zip(kt_means, kt_stds)],
    [m + s for m, s in zip(kt_means, kt_stds)],
    alpha=0.2, color="#9467bd"
)
axes[1, 1].set_title("(d) Kendall τ Rank Correlation")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Kendall τ")

fig.suptitle("GNN Betweenness Centrality — Training Results", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_training_grid.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_training_grid.png  (combined 2x2 for paper Figure 4.1)")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — PRINT FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
final_train_loss  = train_losses[-1]
final_test_loss   = test_losses[-1]
final_kt_mean     = kt_means[-1]
final_kt_std      = kt_stds[-1]
final_rel_diff    = rel_diffs[-1]

print(f"\n{'='*60}")
print(f"  FINAL RESULTS (Epoch {EPOCHS})")
print(f"{'='*60}")
print(f"  Training Loss        : {final_train_loss:.4f}")
print(f"  Test Loss            : {final_test_loss:.4f}")
print(f"  Relative Difference  : {final_rel_diff:.4f}")
print(f"  Kendall tau (mean)   : {final_kt_mean:.4f}")
print(f"  Kendall tau (std)    : {final_kt_std:.4f}")
print(f"{'='*60}")
print(f"\n  All figures saved to: {OUTPUT_DIR}/")
print(f"  Files generated:")
print(f"    fig_training_loss.png         — use as Figure 4.1(a)")
print(f"    fig_test_loss.png             — use as Figure 4.1(b)")
print(f"    fig_relative_loss_diff.png    — use as Figure 4.1(c)")
print(f"    fig_kendall_tau.png           — use as Figure 4.1(d)")
print(f"    fig_training_grid.png         — combined 2x2 for Figure 4.1")
print(f"    fig_betweenness_comparison.png — use as Figure 4.2")
print(f"\n  Copy these into your paper at the locations marked in Section 4.6\n")

# ─ Save numerical results to text file ───────────────────────────────────────
results_path = f"{OUTPUT_DIR}/results_summary.txt"
with open(results_path, "w") as f:
    f.write("GNN Betweenness Centrality — Experiment Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Settings:\n")
    f.write(f"  Nodes         : {GRAPH_NODES}\n")
    f.write(f"  Total graphs  : {TOTAL_GRAPHS}\n")
    f.write(f"  Train graphs  : {len(adj_train)}\n")
    f.write(f"  Test graphs   : {len(adj_test)}\n")
    f.write(f"  Epochs        : {EPOCHS}\n")
    f.write(f"  Learning rate : {LEARNING_RATE}\n")
    f.write(f"  Dropout       : {DROPOUT}\n")
    f.write(f"  Weight decay  : {WEIGHT_DECAY}\n\n")
    f.write(f"Final Results:\n")
    f.write(f"  Training Loss : {final_train_loss:.6f}\n")
    f.write(f"  Test Loss     : {final_test_loss:.6f}\n")
    f.write(f"  Rel Diff      : {final_rel_diff:.6f}\n")
    f.write(f"  Kendall tau   : {final_kt_mean:.6f} ± {final_kt_std:.6f}\n\n")
    f.write("Per-epoch results:\n")
    f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Test Loss':>12} {'Rel Diff':>10} {'KT Mean':>10} {'KT Std':>10}\n")
    for e in range(EPOCHS):
        f.write(f"{e+1:>6} {train_losses[e]:>12.6f} {test_losses[e]:>12.6f} "
                f"{rel_diffs[e]:>10.6f} {kt_means[e]:>10.6f} {kt_stds[e]:>10.6f}\n")

print(f"  Numerical results saved to: {results_path}")
