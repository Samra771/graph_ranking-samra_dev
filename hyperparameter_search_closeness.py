"""
hyperparameter_search_closeness.py
====================================
Grid search over hyperparameters for the CLOSENESS GNN.

Identical structure to the betweenness grid search but:
- Uses undirected ER graphs (directed graphs cause NaN for closeness)
- Uses degree-normalised adjacency (D^-1 A) as second input
- Uses GNN_Close model instead of GNN_Bet
- Uses closeness_centrality as ground truth labels

Run:
    python hyperparameter_search_closeness.py

Expected runtime: 6-8 hours on CPU 

Output files:
    hyperparameter_search_closeness_results.csv
    best_hyperparameters_closeness.json
    hyperparameter_heatmap_closeness.png
    hyperparameter_search_closeness_log.txt
"""

import os
import json
import csv
import time
import itertools
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.sparse as sp

from closeness_model import GNN_Close
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation
)

# ═════════════════════════════════════════════════════════════════════
# FIXED SETTINGS
# ═════════════════════════════════════════════════════════════════════
GRAPH_NODES      = 200
GRAPH_SPARSENESS = 0.15
TOTAL_GRAPHS     = 500
TRAIN_RATIO      = 0.70    # 350 training graphs
VAL_RATIO        = 0.15    # 75 validation graphs
SEED             = 42
EPOCHS_SEARCH    = 30
BATCH_SIZE       = 16
MODEL_SIZE       = GRAPH_NODES

# ═════════════════════════════════════════════════════════════════════
# HYPERPARAMETER GRID — 18 combinations
# ═════════════════════════════════════════════════════════════════════
GRID = {
    "learning_rate" : [1e-3, 5e-4, 1e-4],
    "dropout"       : [0.2, 0.4, 0.6],
    "hidden_size"   : [10, 20, 40],
    "weight_decay"  : [0.0, 0.01],
}

all_configs = list(itertools.product(
    GRID["learning_rate"],
    GRID["dropout"],
    GRID["hidden_size"],
    GRID["weight_decay"],
))

# ═════════════════════════════════════════════════════════════════════
# SETUP
# ═════════════════════════════════════════════════════════════════════
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*60}")
print(f"  Hyperparameter Grid Search — GNN Closeness")
print(f"{'='*60}")
print(f"  Device          : {device}")
print(f"  Total configs   : {len(all_configs)}")
print(f"  Epochs per run  : {EPOCHS_SEARCH}")
print(f"  Total graphs    : {TOTAL_GRAPHS}")
print(f"  Train / Val     : {int(TOTAL_GRAPHS*TRAIN_RATIO)} / "
      f"{int(TOTAL_GRAPHS*VAL_RATIO)}")
print(f"{'='*60}\n")

# ═════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════

def make_undirected_er(n, p):
    """Undirected ER graph — required for closeness centrality."""
    return nx.erdos_renyi_graph(n, p, directed=False)

def get_adj(g, ms):
    """Standard adjacency matrix padded to model size."""
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")
    if n < ms:
        adj = sp.block_diag([adj, csr_matrix((ms-n, ms-n))])
    return adj.tocsr()

def get_degree_normalised_adj(g, ms):
    """
    Degree-normalised adjacency A_mod = D^{-1} A.
    Prevents zero feature vectors which cause NaN during training.
    """
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")

    degrees          = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0   # safety — avoid division by zero
    D_inv            = sp.diags(1.0 / degrees)
    adj_mod          = D_inv @ adj

    if n < ms:
        adj_mod = sp.block_diag([adj_mod, csr_matrix((ms-n, ms-n))])
    return adj_mod.tocsr()

def to_array(d, n):
    arr = np.zeros(n, dtype=np.float32)
    for node, val in d.items():
        if 0 <= node < n:
            arr[node] = float(val)
    return arr

# ═════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE DATASET ONCE
# ═════════════════════════════════════════════════════════════════════
print("STEP 1: Generating graphs...")
print("        Using undirected ER graphs for closeness centrality")
print()

t0 = time.time()
adjs, adj_mods, cents, ns = [], [], [], []

for i in range(TOTAL_GRAPHS):
    g      = make_undirected_er(GRAPH_NODES, GRAPH_SPARSENESS)
    n      = g.number_of_nodes()
    adj    = get_adj(g, MODEL_SIZE)
    adj_mod= get_degree_normalised_adj(g, MODEL_SIZE)
    cc     = to_array(nx.closeness_centrality(g), n)

    adjs.append(adj)
    adj_mods.append(adj_mod)
    cents.append(cc)
    ns.append(n)

    if (i+1) % 100 == 0:
        print(f"  {i+1}/{TOTAL_GRAPHS} graphs generated")

print(f"  Done in {time.time()-t0:.0f}s\n")

# ── Train / val split ─────────────────────────────────────────────────
idx      = list(range(TOTAL_GRAPHS))
np.random.shuffle(idx)
n_train  = int(TOTAL_GRAPHS * TRAIN_RATIO)
n_val    = int(TOTAL_GRAPHS * VAL_RATIO)

train_idx = idx[:n_train]
val_idx   = idx[n_train:n_train+n_val]

tr_a   = [adjs[i]     for i in train_idx]
tr_am  = [adj_mods[i] for i in train_idx]
tr_c   = [cents[i]    for i in train_idx]
tr_n   = [ns[i]       for i in train_idx]

va_a   = [adjs[i]     for i in val_idx]
va_am  = [adj_mods[i] for i in val_idx]
va_c   = [cents[i]    for i in val_idx]
va_n   = [ns[i]       for i in val_idx]

print(f"  Train: {len(tr_a)} graphs | Val: {len(va_a)} graphs\n")

# ═════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN AND EVALUATE ONE CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

def train_and_evaluate(lr, dropout, hidden, weight_decay):
    """Train closeness GNN and return validation Kendall tau."""

    net      = GNN_Close(
        ninput       = MODEL_SIZE,
        nhid         = hidden,
        dropout      = dropout,
        learning_rate= lr,
        weight_decay = weight_decay
    )
    model, _ = net.model_to_device(net)
    model    = model.to(device)
    opt      = net.get_optimizer(model)

    # Training loop
    model.train()
    for epoch in range(EPOCHS_SEARCH):
        perm = np.random.permutation(len(tr_a))
        for batch_start in range(0, len(tr_a), BATCH_SIZE):
            batch = perm[batch_start:batch_start+BATCH_SIZE]
            opt.zero_grad()
            bloss = torch.tensor(
                        0.0, device=device, requires_grad=True)
            cnt = 0
            for i in batch:
                adj_t  = sparse_mx_to_torch_sparse_tensor(
                             tr_a[i],  device)
                adj_mt = sparse_mx_to_torch_sparse_tensor(
                             tr_am[i], device)
                y = model(adj_t, adj_mt)
                if torch.isnan(y).any():
                    continue
                tv    = torch.from_numpy(
                            tr_c[i]).float().to(device)
                loss  = loss_cal(
                            y, tv, tr_n[i], device, MODEL_SIZE)
                bloss = bloss + loss
                cnt  += 1
            if cnt > 0:
                bloss = bloss / cnt
                bloss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)
                opt.step()

    # Validation
    model.eval()
    kt_scores = []
    with torch.no_grad():
        for i in range(len(va_a)):
            adj_t  = sparse_mx_to_torch_sparse_tensor(
                         va_a[i],  device)
            adj_mt = sparse_mx_to_torch_sparse_tensor(
                         va_am[i], device)
            y = model(adj_t, adj_mt)
            if torch.isnan(y).any():
                continue
            tv = torch.from_numpy(va_c[i]).float().to(device)
            kt = ranking_correlation(y, tv, va_n[i], MODEL_SIZE)
            if not np.isnan(kt):
                kt_scores.append(kt)

    mean_kt = float(np.mean(kt_scores)) if kt_scores else 0.0
    std_kt  = float(np.std(kt_scores))  if kt_scores else 0.0
    return mean_kt, std_kt

# ═════════════════════════════════════════════════════════════════════
# STEP 3 — RUN ALL 18 CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════
print("STEP 2: Running grid search...\n")

results     = []
best_kt     = -1
best_config = None
total_t0    = time.time()

csv_path = "./hyperparameter_search_closeness_results.csv"
log_path = "./hyperparameter_search_closeness_log.txt"

with open(csv_path, "w", newline="", encoding="utf-8") as csvf, \
     open(log_path, "w", encoding="utf-8") as logf:

    writer = csv.writer(csvf)
    writer.writerow([
        "config_num", "learning_rate", "dropout",
        "hidden_size", "weight_decay",
        "val_kt_mean", "val_kt_std", "time_seconds"
    ])

    logf.write("Closeness Hyperparameter Grid Search Log\n")
    logf.write("="*50 + "\n\n")

    for idx_c, (lr, dropout, hidden, wd) in enumerate(all_configs):

        elapsed_total  = time.time() - total_t0
        avg_per_config = elapsed_total / max(1, idx_c)
        est_remaining  = avg_per_config * (len(all_configs) - idx_c)

        print(f"  [{idx_c+1:2d}/{len(all_configs)}] "
              f"lr={lr:.0e}  dropout={dropout}  "
              f"hidden={hidden:2d}  wd={wd}  "
              f"| est. remaining: "
              f"{est_remaining/60:.0f}min")

        t_config        = time.time()
        kt_mean, kt_std = train_and_evaluate(
                              lr, dropout, hidden, wd)
        elapsed_config  = time.time() - t_config

        print(f"           Val KT = {kt_mean:.4f} "
              f"+/- {kt_std:.4f}  "
              f"({elapsed_config:.0f}s)")

        writer.writerow([
            idx_c+1, lr, dropout, hidden, wd,
            f"{kt_mean:.6f}", f"{kt_std:.6f}",
            f"{elapsed_config:.1f}"
        ])
        csvf.flush()

        logf.write(
            f"Config {idx_c+1}: "
            f"lr={lr}, dropout={dropout}, "
            f"hidden={hidden}, wd={wd}\n"
            f"  Val KT = {kt_mean:.6f} "
            f"+/- {kt_std:.6f} "
            f"({elapsed_config:.0f}s)\n\n"
        )
        logf.flush()

        result = {
            "config_num"   : idx_c + 1,
            "learning_rate": lr,
            "dropout"      : dropout,
            "hidden_size"  : hidden,
            "weight_decay" : wd,
            "val_kt_mean"  : kt_mean,
            "val_kt_std"   : kt_std,
        }
        results.append(result)

        if kt_mean > best_kt:
            best_kt     = kt_mean
            best_config = result.copy()
            print(f"           *** NEW BEST ***")

    logf.write(f"\nBEST CONFIGURATION:\n")
    logf.write(json.dumps(best_config, indent=2))

print(f"\n  Grid search complete. "
      f"Total time: "
      f"{(time.time()-total_t0)/3600:.2f} hours\n")

# ═════════════════════════════════════════════════════════════════════
# STEP 4 — SAVE BEST CONFIGURATION
# ═════════════════════════════════════════════════════════════════════
with open("./best_hyperparameters_closeness.json", "w") as f:
    json.dump(best_config, f, indent=2)

print(f"{'='*60}")
print(f"  BEST CONFIGURATION — CLOSENESS")
print(f"{'='*60}")
print(f"  Learning rate : {best_config['learning_rate']}")
print(f"  Dropout       : {best_config['dropout']}")
print(f"  Hidden size   : {best_config['hidden_size']}")
print(f"  Weight decay  : {best_config['weight_decay']}")
print(f"  Val KT        : {best_config['val_kt_mean']:.4f} "
      f"+/- {best_config['val_kt_std']:.4f}")
print(f"{'='*60}\n")
print(f"  All results : {csv_path}")
print(f"  Best config : ./best_hyperparameters_closeness.json\n")

# ═════════════════════════════════════════════════════════════════════
# STEP 5 — HEATMAP FIGURE
# ═════════════════════════════════════════════════════════════════════
print("STEP 3: Generating heatmap...")

lrs      = GRID["learning_rate"]
dropouts = GRID["dropout"]

matrix = np.zeros((len(lrs), len(dropouts)))
for r in results:
    if r["hidden_size"] == 20 and r["weight_decay"] == 0.01:
        if r["learning_rate"] in lrs and r["dropout"] in dropouts:
            row = lrs.index(r["learning_rate"])
            col = dropouts.index(r["dropout"])
            matrix[row, col] = r["val_kt_mean"]

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Validation Kendall tau")

ax.set_xticks(range(len(dropouts)))
ax.set_xticklabels([f"{d}" for d in dropouts])
ax.set_yticks(range(len(lrs)))
ax.set_yticklabels([f"{lr:.0e}" for lr in lrs])
ax.set_xlabel("Dropout probability")
ax.set_ylabel("Learning rate")
ax.set_title(
    "Closeness Grid Search: Validation Kendall tau\n"
    "(hidden size = 20, weight decay = 0.01)"
)

for i in range(len(lrs)):
    for j in range(len(dropouts)):
        val   = matrix[i, j]
        color = "white" if val < 0.5 else "black"
        ax.text(j, i, f"{val:.3f}",
                ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

fig.tight_layout()
fig.savefig("./hyperparameter_heatmap_closeness.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: hyperparameter_heatmap_closeness.png\n")

