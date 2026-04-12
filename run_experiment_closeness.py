"""
run_experiment_closeness.py  (FIXED VERSION)
============================================
Fixes applied vs previous version:

Problem 1 — NaN loss:
  The column-masked adjacency (adj_mod) was producing zero-vectors in many
  GNN layers because sparse directed graphs have many zero-degree nodes.
  L2 normalisation of a zero-vector produces NaN (0/0).
  Fix: use undirected ER graphs for closeness. Every node in a connected
  undirected graph has nonzero closeness. adj_mod is replaced by a
  degree-normalised version of A that cannot produce zero vectors.

Problem 2 — Loss stuck at 1.0:
  When GNN output is constant (same value for all nodes), MarginRankingLoss
  returns exactly 1.0 and gradients vanish. This happened because NaN in
  the forward pass was being handled as 0 by PyTorch, producing flat output.
  Fix: resolved by fixing Problem 1.

Problem 3 — Directed graph closeness zeros:
  In sparse directed ER graphs many nodes cannot reach others, giving
  closeness=0 for a large fraction of nodes. This makes ranking degenerate.
  Fix: undirected graphs have well-defined nonzero closeness for all nodes
  in the largest connected component, which is almost always the whole graph
  for ER with p=0.15 and N=200.
"""

import os
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import time
import random

# ── Import repo files ─────────────────────────────────────────────────────────
from closeness_model import GNN_Close
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
GRAPH_NODES      = 200
GRAPH_SPARSENESS = 0.15     # p=0.15 gives dense enough undirected graphs
TOTAL_GRAPHS     = 5000
TRAIN_RATIO      = 0.80

HIDDEN_LAYERS    = 20
EPOCHS           = 50
LEARNING_RATE    = 5e-4
DROPOUT          = 0.2
WEIGHT_DECAY     = 0.01
BATCH_SIZE       = 16

MODEL_SIZE       = GRAPH_NODES
OUTPUT_DIR       = "./paper_figures_closeness"
MODEL_SAVE_PATH  = "./closeness_model.pth"

# ═════════════════════════════════════════════════════════════════════════════
# SETUP
# ═════════════════════════════════════════════════════════════════════════════
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*60}")
print(f"  GNN Closeness Centrality Experiment (FIXED)")
print(f"{'='*60}")
print(f"  Device        : {device}")
print(f"  Graph type    : Undirected Erdos-Renyi")
print(f"  Nodes         : {GRAPH_NODES}")
print(f"  Graphs        : {TOTAL_GRAPHS}")
print(f"  Epochs        : {EPOCHS}")
print(f"  Output dir    : {OUTPUT_DIR}")
print(f"{'='*60}\n")

# ═════════════════════════════════════════════════════════════════════════════
# GRAPH GENERATION FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def create_undirected_er_graph(n, p):
    """
    Creates an undirected Erdos-Renyi graph.
    Why undirected for closeness?
      - Every node has well-defined nonzero closeness
      - No isolated nodes in the reachability sense
      - Closeness values spread across [0,1] giving good ranking diversity
    """
    return nx.erdos_renyi_graph(n, p, directed=False)


def get_adjacency(graph, model_size):
    """
    Returns the adjacency matrix padded to model_size x model_size.
    Padding with zeros allows all graphs to have the same tensor size.
    """
    n = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype(np.float32)

    # Pad to model_size if needed
    if n < model_size:
        adj = sp.block_diag([adj, csr_matrix((model_size - n, model_size - n))])

    return adj.tocsr()


def get_degree_normalised_adjacency(graph, model_size):
    """
    Builds adj_mod — the second input to the closeness GNN.

    Instead of column masking (which caused NaN), we use
    degree-normalised adjacency: A_mod = D^{-1} A
    where D is the diagonal degree matrix.

    This:
      - Can never produce zero vectors (every node has degree > 0
        in a connected undirected ER graph with p=0.15, N=200)
      - Properly encodes the local connectivity structure
      - Is standard in graph convolution literature (Kipf & Welling 2017)
    """
    n = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype(np.float32)

    # Compute degree vector and its inverse
    degrees = np.array(adj.sum(axis=1)).flatten()
    # Replace zero degrees with 1 to avoid division by zero (safety)
    degrees[degrees == 0] = 1.0
    d_inv   = 1.0 / degrees
    D_inv   = sp.diags(d_inv)

    # Degree-normalised adjacency
    adj_mod = D_inv @ adj

    # Pad to model_size
    if n < model_size:
        adj_mod = sp.block_diag([adj_mod, csr_matrix((model_size - n, model_size - n))])

    return adj_mod.tocsr()


def get_closeness(graph):
    """
    Computes exact closeness centrality using NetworkX.
    For undirected graphs this is well-defined for all nodes.
    Returns dict {node_id: value}.
    """
    return nx.closeness_centrality(graph)


def cc_dict_to_array(cc_dict, num_nodes):
    """Convert closeness dict to numpy array of length num_nodes."""
    arr = np.zeros(num_nodes, dtype=np.float32)
    for node, val in cc_dict.items():
        if 0 <= node < num_nodes:
            arr[node] = float(val)
    return arr

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE GRAPHS
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 1: Generating graphs...")

adj_matrices      = []
adj_matrices_mod  = []
closeness_list    = []
node_counts       = []

for i in range(TOTAL_GRAPHS):
    graph   = create_undirected_er_graph(GRAPH_NODES, GRAPH_SPARSENESS)
    adj     = get_adjacency(graph, MODEL_SIZE)
    adj_mod = get_degree_normalised_adjacency(graph, MODEL_SIZE)
    cc      = get_closeness(graph)
    n       = graph.number_of_nodes()

    adj_matrices.append(adj)
    adj_matrices_mod.append(adj_mod)
    closeness_list.append(cc)
    node_counts.append(n)

    if (i + 1) % 500 == 0:
        print(f"  Generated {i+1}/{TOTAL_GRAPHS} graphs")

# ── Sanity check: print closeness range on first graph ────────────────────────
sample_cc  = np.array(list(closeness_list[0].values()))
print(f"\n  Sanity check on first graph:")
print(f"    Closeness min  : {sample_cc.min():.4f}")
print(f"    Closeness max  : {sample_cc.max():.4f}")
print(f"    Closeness mean : {sample_cc.mean():.4f}")
print(f"    Zero values    : {(sample_cc == 0).sum()} / {len(sample_cc)}")
print(f"    (Zero values should be 0 or very few for undirected ER)")

# ── Train / test split ────────────────────────────────────────────────────────
indices       = list(range(TOTAL_GRAPHS))
random.shuffle(indices)
train_size    = int(TOTAL_GRAPHS * TRAIN_RATIO)
train_indices = set(indices[:train_size])
test_indices  = set(indices[train_size:])

adj_train     = [adj_matrices[i]     for i in range(TOTAL_GRAPHS) if i in train_indices]
adj_mod_train = [adj_matrices_mod[i] for i in range(TOTAL_GRAPHS) if i in train_indices]
cc_train      = [closeness_list[i]   for i in range(TOTAL_GRAPHS) if i in train_indices]
nodes_train   = [node_counts[i]      for i in range(TOTAL_GRAPHS) if i in train_indices]

adj_test      = [adj_matrices[i]     for i in range(TOTAL_GRAPHS) if i in test_indices]
adj_mod_test  = [adj_matrices_mod[i] for i in range(TOTAL_GRAPHS) if i in test_indices]
cc_test       = [closeness_list[i]   for i in range(TOTAL_GRAPHS) if i in test_indices]
nodes_test    = [node_counts[i]      for i in range(TOTAL_GRAPHS) if i in test_indices]

print(f"\n  Train: {len(adj_train)} | Test: {len(adj_test)}\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD MODEL
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 2: Building model...")

network   = GNN_Close(
    ninput=MODEL_SIZE,
    nhid=HIDDEN_LAYERS,
    dropout=DROPOUT,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
model, _  = network.model_to_device(network)
model     = model.to(device)
optimizer = network.get_optimizer(model)
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(adj_list, adj_mod_list, cc_dicts, node_list):
    model.train()
    total_loss  = 0.0
    num_samples = len(adj_list)
    perm        = np.random.permutation(num_samples)

    for batch_start in range(0, num_samples, BATCH_SIZE):
        batch_idx = perm[batch_start : batch_start + BATCH_SIZE]
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0

        for i in batch_idx:
            adj     = sparse_mx_to_torch_sparse_tensor(adj_list[i],     device)
            adj_mod = sparse_mx_to_torch_sparse_tensor(adj_mod_list[i], device)
            n       = node_list[i]

            y_out    = model(adj, adj_mod)

            # Check for NaN in output — skip batch if found
            if torch.isnan(y_out).any():
                print(f"  WARNING: NaN detected in forward pass at graph {i} — skipping")
                continue

            true_arr = cc_dict_to_array(cc_dicts[i], MODEL_SIZE)
            true_val = torch.from_numpy(true_arr).float().to(device)

            loss = loss_cal(y_out, true_val, n, device, MODEL_SIZE)
            batch_loss = batch_loss + loss
            count += 1

        if count > 0:
            batch_loss = batch_loss / count
            batch_loss.backward()
            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(batch_loss.detach())

    return total_loss / max(1, num_samples // BATCH_SIZE)


def evaluate(adj_list, adj_mod_list, cc_dicts, node_list):
    model.eval()
    total_loss = 0.0
    kt_scores  = []

    with torch.no_grad():
        for i in range(len(adj_list)):
            adj     = sparse_mx_to_torch_sparse_tensor(adj_list[i],     device)
            adj_mod = sparse_mx_to_torch_sparse_tensor(adj_mod_list[i], device)
            n       = node_list[i]

            y_out    = model(adj, adj_mod)

            if torch.isnan(y_out).any():
                continue

            true_arr = cc_dict_to_array(cc_dicts[i], MODEL_SIZE)
            true_val = torch.from_numpy(true_arr).float().to(device)

            loss = loss_cal(y_out, true_val, n, device, MODEL_SIZE)
            total_loss += float(loss)

            kt = ranking_correlation(y_out, true_val, n, MODEL_SIZE)
            if not np.isnan(kt):
                kt_scores.append(kt)

    avg_loss = total_loss / max(1, len(adj_list))
    avg_kt   = float(np.mean(kt_scores)) if kt_scores else 0.0
    std_kt   = float(np.std(kt_scores))  if kt_scores else 0.0
    return avg_loss, avg_kt, std_kt

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — TRAIN
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 3: Training...")

train_losses = []
test_losses  = []
rel_diffs    = []
kt_means     = []
kt_stds      = []

t0 = time.time()

for epoch in range(EPOCHS):
    tr_loss              = train_one_epoch(adj_train, adj_mod_train, cc_train, nodes_train)
    te_loss, kt_m, kt_s = evaluate(adj_test, adj_mod_test, cc_test, nodes_test)
    rel_diff             = (tr_loss - te_loss) / max(abs(tr_loss), 1e-9)

    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    rel_diffs.append(rel_diff)
    kt_means.append(kt_m)
    kt_stds.append(kt_s)

    if (epoch + 1) % 5 == 0:
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {tr_loss:.4f} | "
              f"Test: {te_loss:.4f} | "
              f"KT: {kt_m:.4f} ± {kt_s:.4f} | "
              f"Time: {elapsed:.0f}s")

print(f"\n  Done. Total time: {time.time()-t0:.0f}s")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"  Model saved: {MODEL_SAVE_PATH}\n")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — GENERATE FIGURES
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 4: Generating figures...")

STYLE = {
    "figure.dpi"     : 300,
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 12,
    "axes.labelsize" : 11,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "axes.grid"      : True,
    "grid.alpha"     : 0.3,
}
plt.rcParams.update(STYLE)
epochs_range = range(1, EPOCHS + 1)

# ── Individual figures ────────────────────────────────────────────────────────
for data, color, ylabel, title, fname in [
    (train_losses, "#1f77b4", "Loss",    "Training Loss vs Epoch (Closeness)",  "fig_training_loss.png"),
    (test_losses,  "#ff7f0e", "Loss",    "Test Loss vs Epoch (Closeness)",      "fig_test_loss.png"),
    (rel_diffs,    "#2ca02c", "(Train-Test)/Train", "Relative Loss Difference", "fig_relative_loss_diff.png"),
]:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs_range, data, color=color)
    if fname == "fig_relative_loss_diff.png":
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")

# Kendall tau with shaded std band
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
ax.set_title("Kendall Rank Correlation vs Epoch (Closeness Centrality)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_kendall_tau.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_kendall_tau.png")

# ── Combined 2x2 grid ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(epochs_range, train_losses, color="#1f77b4")
axes[0,0].set_title("(a) Training Loss vs Epoch")
axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")

axes[0,1].plot(epochs_range, test_losses, color="#ff7f0e")
axes[0,1].set_title("(b) Test Loss vs Epoch")
axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Loss")

axes[1,0].plot(epochs_range, rel_diffs, color="#2ca02c")
axes[1,0].axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axes[1,0].set_title("(c) Relative Loss Difference")
axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("(Train − Test) / Train")

axes[1,1].plot(epochs_range, kt_means, color="#9467bd")
axes[1,1].fill_between(
    epochs_range,
    [m-s for m,s in zip(kt_means, kt_stds)],
    [m+s for m,s in zip(kt_means, kt_stds)],
    alpha=0.2, color="#9467bd"
)
axes[1,1].set_title("(d) Kendall τ Rank Correlation")
axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("Kendall τ")

fig.suptitle("GNN Closeness Centrality — Training Results", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_training_grid.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_training_grid.png")

# ── Comparison plot: GNN vs NetworkX ─────────────────────────────────────────
model.eval()
kt_final = []
predictions_all   = []
ground_truths_all = []

with torch.no_grad():
    for i in range(len(adj_test)):
        adj     = sparse_mx_to_torch_sparse_tensor(adj_test[i],     device)
        adj_mod = sparse_mx_to_torch_sparse_tensor(adj_mod_test[i], device)
        n       = nodes_test[i]

        y_out    = model(adj, adj_mod)
        if torch.isnan(y_out).any():
            kt_final.append(np.nan)
            continue

        true_arr = cc_dict_to_array(cc_test[i], MODEL_SIZE)
        true_val = torch.from_numpy(true_arr).float().to(device)

        kt = ranking_correlation(y_out, true_val, n, MODEL_SIZE)
        kt_final.append(kt)
        predictions_all.append(y_out.cpu().numpy().flatten()[:n])
        ground_truths_all.append(true_arr[:n])

# Pick graph closest to median Kendall tau
kt_array  = np.array([k for k in kt_final if not np.isnan(k)])
valid_idx  = [i for i, k in enumerate(kt_final) if not np.isnan(k)]
median_kt  = np.median(kt_array)
best_local = int(np.argmin(np.abs(kt_array - median_kt)))
best_idx   = valid_idx[best_local]

pred_plot  = predictions_all[best_local]
truth_plot = ground_truths_all[best_local]
n_plot     = nodes_test[best_idx]

def normalise(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx - mn > 1e-9 else arr

node_idx = np.arange(n_plot)
fig, ax  = plt.subplots(figsize=(8, 4))
ax.plot(node_idx, normalise(truth_plot),
        "r--o", markersize=3, linewidth=1.0, label="NetworkX (Ground Truth)")
ax.plot(node_idx, normalise(pred_plot),
        "b--o", markersize=3, linewidth=1.0, label="GNN Prediction")
ax.set_xlabel("Node Index")
ax.set_ylabel("Closeness Centrality (normalised)")
ax.set_title(f"Closeness Centrality: GNN vs Ground Truth  (Kendall τ = {kt_array[best_local]:.3f})")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_closeness_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_closeness_comparison.png  (τ = {kt_array[best_local]:.3f})")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  FINAL RESULTS — Closeness Centrality (Epoch {EPOCHS})")
print(f"{'='*60}")
print(f"  Training Loss       : {train_losses[-1]:.4f}")
print(f"  Test Loss           : {test_losses[-1]:.4f}")
print(f"  Relative Difference : {rel_diffs[-1]:.4f}")
print(f"  Kendall tau (mean)  : {kt_means[-1]:.4f}")
print(f"  Kendall tau (std)   : {kt_stds[-1]:.4f}")
print(f"{'='*60}")
print(f"\n  Figures saved to: {OUTPUT_DIR}/")
print(f"    fig_training_grid.png        → paper Figure 4.3")
print(f"    fig_closeness_comparison.png → paper Figure 4.4\n")

# Save numerical results
with open(f"{OUTPUT_DIR}/results_summary.txt", "w") as f:
    f.write("GNN Closeness Centrality — Experiment Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Graph type    : Undirected Erdos-Renyi\n")
    f.write(f"Nodes         : {GRAPH_NODES}\n")
    f.write(f"Total graphs  : {TOTAL_GRAPHS}\n")
    f.write(f"Train / Test  : {len(adj_train)} / {len(adj_test)}\n")
    f.write(f"Epochs        : {EPOCHS}\n")
    f.write(f"Learning rate : {LEARNING_RATE}\n")
    f.write(f"Dropout       : {DROPOUT}\n")
    f.write(f"Weight decay  : {WEIGHT_DECAY}\n\n")
    f.write(f"Final Results:\n")
    f.write(f"  Training Loss : {train_losses[-1]:.6f}\n")
    f.write(f"  Test Loss     : {test_losses[-1]:.6f}\n")
    f.write(f"  Rel Diff      : {rel_diffs[-1]:.6f}\n")
    f.write(f"  Kendall tau   : {kt_means[-1]:.6f} ± {kt_stds[-1]:.6f}\n\n")
    f.write(f"{'Epoch':>6} {'Train':>12} {'Test':>12} {'RelDiff':>10} {'KT Mean':>10} {'KT Std':>10}\n")
    for e in range(EPOCHS):
        f.write(f"{e+1:>6} {train_losses[e]:>12.6f} {test_losses[e]:>12.6f} "
                f"{rel_diffs[e]:>10.6f} {kt_means[e]:>10.6f} {kt_stds[e]:>10.6f}\n")
print(f"  Results saved: {OUTPUT_DIR}/results_summary.txt")