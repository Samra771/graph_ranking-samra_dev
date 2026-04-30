"""
hyperparameter_search.py
=========================
Grid search over hyperparameters for the betweenness GNN.

What this script does:
- Tries 18 different combinations of hyperparameters
- Trains a model for each combination (30 epochs each)
- Evaluates each on a validation set using Kendall tau
- Saves all results to hyperparameter_search_results.csv
- Identifies and saves the best configuration
- Generates a heatmap figure showing all results

Run:
    python hyperparameter_search.py

Expected runtime: 6-8 hours on CPU (runs overnight)

Output files:
    hyperparameter_search_results.csv  -- all 18 results
    best_hyperparameters.json          -- best configuration
    hyperparameter_heatmap.png         -- visual results
    hyperparameter_search_log.txt      -- full log
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

from betweennes_model import GNN_Bet
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation
)

# ═════════════════════════════════════════════════════════════════════
# FIXED SETTINGS — do not change these
# ═════════════════════════════════════════════════════════════════════
GRAPH_NODES      = 200
GRAPH_SPARSENESS = 0.15
TOTAL_GRAPHS     = 500     # enough for reliable validation
TRAIN_RATIO      = 0.70    # 350 training graphs
VAL_RATIO        = 0.15    # 75 validation graphs (used for selection)
TEST_RATIO       = 0.15    # 75 test graphs (not used here)
SEED             = 42
EPOCHS_SEARCH    = 30      # quick training per configuration
BATCH_SIZE       = 16
MODEL_SIZE       = GRAPH_NODES

# ═════════════════════════════════════════════════════════════════════
# HYPERPARAMETER GRID — 18 total combinations
# ═════════════════════════════════════════════════════════════════════
GRID = {
    "learning_rate" : [1e-3, 1e-4, 5e-5],
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
print(f"  Hyperparameter Grid Search — GNN Betweenness")
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

def make_er_graph(n, p):
    return nx.erdos_renyi_graph(n, p, directed=False)

def get_adj(g, ms):
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")
    if n < ms:
        adj = sp.block_diag([adj, csr_matrix((ms-n, ms-n))])
    return adj.tocsr()

def get_adj_T(g, ms):
    return get_adj(g, ms).T.tocsr()

def to_array(d, n):
    arr = np.zeros(n, dtype=np.float32)
    for node, val in d.items():
        if 0 <= node < n:
            arr[node] = float(val)
    return arr

# ═════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE DATASET ONCE
# Used for all 18 configurations
# ═════════════════════════════════════════════════════════════════════
print("STEP 1: Generating graphs (done once for all configurations)...")
t0 = time.time()

adjs, adj_Ts, cents, ns = [], [], [], []

for i in range(TOTAL_GRAPHS):
    g   = make_er_graph(GRAPH_NODES, GRAPH_SPARSENESS)
    n   = g.number_of_nodes()
    adj = get_adj(g, MODEL_SIZE)
    adjT= get_adj_T(g, MODEL_SIZE)
    bc  = to_array(
              nx.betweenness_centrality(g, normalized=True), n)

    adjs.append(adj)
    adj_Ts.append(adjT)
    cents.append(bc)
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

tr_a  = [adjs[i]   for i in train_idx]
tr_aT = [adj_Ts[i] for i in train_idx]
tr_c  = [cents[i]  for i in train_idx]
tr_n  = [ns[i]     for i in train_idx]

va_a  = [adjs[i]   for i in val_idx]
va_aT = [adj_Ts[i] for i in val_idx]
va_c  = [cents[i]  for i in val_idx]
va_n  = [ns[i]     for i in val_idx]

print(f"  Train: {len(tr_a)} graphs | Val: {len(va_a)} graphs\n")

# ═════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN AND EVALUATE ONE CONFIGURATION
# ═════════════════════════════════════════════════════════════════════

def train_and_evaluate(lr, dropout, hidden, weight_decay):
    """
    Trains GNN with given hyperparameters for EPOCHS_SEARCH epochs.
    Returns mean and std Kendall tau on validation set.
    """
    # Build fresh model for this configuration
    net      = GNN_Bet(
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
            cnt   = 0
            for i in batch:
                adj_t  = sparse_mx_to_torch_sparse_tensor(
                             tr_a[i],  device)
                adjT_t = sparse_mx_to_torch_sparse_tensor(
                             tr_aT[i], device)
                y      = model(adj_t, adjT_t)
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

    # Validation evaluation
    model.eval()
    kt_scores = []
    with torch.no_grad():
        for i in range(len(va_a)):
            adj_t  = sparse_mx_to_torch_sparse_tensor(
                         va_a[i],  device)
            adjT_t = sparse_mx_to_torch_sparse_tensor(
                         va_aT[i], device)
            y      = model(adj_t, adjT_t)
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

results    = []
best_kt    = -1
best_config= None
total_t0   = time.time()

# Open CSV file — writes results as they come in
# So if the job is interrupted you still have partial results
csv_path = "./hyperparameter_search_results.csv"
log_path = "./hyperparameter_search_log.txt"

with open(csv_path, "w", newline="", encoding="utf-8") as csvf, \
     open(log_path, "w", encoding="utf-8") as logf:

    writer = csv.writer(csvf)
    writer.writerow([
        "config_num", "learning_rate", "dropout",
        "hidden_size", "weight_decay",
        "val_kt_mean", "val_kt_std", "time_seconds"
    ])

    logf.write("Hyperparameter Grid Search Log\n")
    logf.write("="*50 + "\n\n")

    for idx_c, (lr, dropout, hidden, wd) in enumerate(all_configs):

        elapsed_total = time.time() - total_t0
        remaining_configs = len(all_configs) - idx_c
        avg_per_config = elapsed_total / max(1, idx_c)
        est_remaining  = avg_per_config * remaining_configs

        print(f"  [{idx_c+1:2d}/{len(all_configs)}] "
              f"lr={lr:.0e}  dropout={dropout}  "
              f"hidden={hidden:2d}  wd={wd}  "
              f"| est. remaining: "
              f"{est_remaining/60:.0f}min")

        t_config = time.time()
        kt_mean, kt_std = train_and_evaluate(lr, dropout, hidden, wd)
        elapsed_config  = time.time() - t_config

        print(f"           Val KT = {kt_mean:.4f} +/- {kt_std:.4f}  "
              f"({elapsed_config:.0f}s)")

        # Write to CSV immediately
        writer.writerow([
            idx_c+1, lr, dropout, hidden, wd,
            f"{kt_mean:.6f}", f"{kt_std:.6f}",
            f"{elapsed_config:.1f}"
        ])
        csvf.flush()

        # Write to log
        logf.write(
            f"Config {idx_c+1}: "
            f"lr={lr}, dropout={dropout}, "
            f"hidden={hidden}, wd={wd}\n"
            f"  Val KT = {kt_mean:.6f} +/- {kt_std:.6f} "
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
      f"Total time: {(time.time()-total_t0)/3600:.2f} hours\n")

# ═════════════════════════════════════════════════════════════════════
# STEP 4 — SAVE BEST CONFIGURATION
# ═════════════════════════════════════════════════════════════════════
with open("./best_hyperparameters.json", "w") as f:
    json.dump(best_config, f, indent=2)

print(f"{'='*60}")
print(f"  BEST CONFIGURATION FOUND")
print(f"{'='*60}")
print(f"  Learning rate : {best_config['learning_rate']}")
print(f"  Dropout       : {best_config['dropout']}")
print(f"  Hidden size   : {best_config['hidden_size']}")
print(f"  Weight decay  : {best_config['weight_decay']}")
print(f"  Val KT        : {best_config['val_kt_mean']:.4f} "
      f"+/- {best_config['val_kt_std']:.4f}")
print(f"{'='*60}\n")
print(f"  All results   : {csv_path}")
print(f"  Best config   : ./best_hyperparameters.json")
print(f"  Full log      : {log_path}\n")

# ═════════════════════════════════════════════════════════════════════
# STEP 5 — GENERATE RESULTS HEATMAP FIGURE
# ═════════════════════════════════════════════════════════════════════
print("STEP 3: Generating results heatmap...")

# Group results by learning rate and dropout
# (fix hidden=20 and wd=0.01 for the main heatmap
#  since those are the most important hyperparameters)
lrs      = GRID["learning_rate"]
dropouts = GRID["dropout"]

# Build matrix — rows = learning rate, cols = dropout
# Using hidden=20 and weight_decay=0.01 slice
matrix = np.zeros((len(lrs), len(dropouts)))
for r in results:
    if r["hidden_size"] == 20 and r["weight_decay"] == 0.01:
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
    "Grid Search: Validation Kendall tau\n"
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
fig.savefig("./hyperparameter_heatmap.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: hyperparameter_heatmap.png\n")

