"""
train_N5000_fixed.py
=====================
FIXED VERSION — corrects the shape mismatch error.

Bug that was fixed:
  cents.append(to_array(bc, n))        # WRONG — n can be < 5000
  cents.append(to_array(bc, MODEL_SIZE)) # CORRECT — always 5000

Upload to Colab and run:
    !python train_N5000_fixed.py
"""

import os
import sys
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
import json
from scipy.sparse import csr_matrix
import scipy.sparse as sp

# ── Check GPU ─────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("ERROR: No GPU.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU    : {gpu_name}")
print(f"Memory : {gpu_mem:.1f} GB\n")

from betweennes_model import GNN_Bet
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation
)

# ═════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════
SEED         = 42
GRAPH_NODES  = 5000
GRAPH_P      = 0.001
TOTAL_GRAPHS = 300
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.10
HIDDEN       = 40
EPOCHS       = 50
LR           = 1e-3
DROPOUT      = 0.4
WD           = 0.01
BATCH_SIZE   = 2
MODEL_SIZE   = GRAPH_NODES   # 5000

OUTPUT_DIR     = "./results_N5000"
MODEL_PATH     = "./results_N5000/betweenness_model_N5000.pth"
CHECKPOINT_DIR = "./results_N5000/checkpoints"

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda")

print("="*55)
print("  GNN Betweenness — N=5000 (FIXED)")
print("="*55)
print(f"  N={GRAPH_NODES}, p={GRAPH_P}")
print(f"  Labels : EXACT betweenness")
print(f"  Graphs : {TOTAL_GRAPHS}")
print(f"  Epochs : {EPOCHS}")
print(f"  LR={LR}, dropout={DROPOUT}, "
      f"hidden={HIDDEN}, wd={WD}")
print("="*55 + "\n")

# ═════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════

def make_graph(n, p):
    g   = nx.erdos_renyi_graph(n, p, directed=False)
    gcc = max(nx.connected_components(g), key=len)
    g   = g.subgraph(gcc).copy()
    return nx.convert_node_labels_to_integers(g)

def get_adj(g, ms):
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")
    if n < ms:
        adj = sp.block_diag([adj, csr_matrix((ms-n, ms-n))])
    return adj.tocsr()

def get_adj_T(g, ms):
    return get_adj(g, ms).T.tocsr()

def to_array(d, n_real, n_padded):
    """
    FIX: always creates array of size n_padded (MODEL_SIZE).
    Fills first n_real entries with centrality values.
    Remaining entries are zero padding.
    This prevents the shape mismatch in loss_cal.
    """
    arr = np.zeros(n_padded, dtype=np.float32)
    for node, val in d.items():
        if 0 <= node < n_real:
            arr[node] = float(val)
    return arr

def get_split(indices, adjs, adj_Ts, cents, ns):
    return ([adjs[i]   for i in indices],
            [adj_Ts[i] for i in indices],
            [cents[i]  for i in indices],
            [ns[i]     for i in indices])

# ═════════════════════════════════════════════════════════════════════
# GENERATE DATASET
# ═════════════════════════════════════════════════════════════════════
print("Generating graphs with EXACT betweenness labels...\n")

adjs, adj_Ts, cents, ns = [], [], [], []
t0 = time.time()

for i in range(TOTAL_GRAPHS):
    g  = make_graph(GRAPH_NODES, GRAPH_P)
    n  = g.number_of_nodes()   # actual nodes (may be < 5000)

    # Exact betweenness
    bc = nx.betweenness_centrality(g, normalized=True)

    adjs.append(get_adj(g,   MODEL_SIZE))
    adj_Ts.append(get_adj_T(g, MODEL_SIZE))

    # FIX: pad to MODEL_SIZE not n
    cents.append(to_array(bc, n, MODEL_SIZE))
    ns.append(n)   # keep actual n for ranking_correlation

    if (i+1) % 5 == 0:
        elapsed   = time.time() - t0
        per_graph = elapsed / (i+1)
        remaining = per_graph * (TOTAL_GRAPHS - i - 1)
        print(f"  [{i+1:3d}/{TOTAL_GRAPHS}] "
              f"{per_graph:.1f}s/graph | "
              f"~{remaining/60:.0f}min remaining")

print(f"\n  Done in {(time.time()-t0)/3600:.2f} hours\n")

# ── Split ─────────────────────────────────────────────────────────────
idx     = list(range(TOTAL_GRAPHS))
random.shuffle(idx)
n_train = int(TOTAL_GRAPHS * TRAIN_RATIO)
n_val   = int(TOTAL_GRAPHS * VAL_RATIO)

tr_a, tr_aT, tr_c, tr_n = get_split(
    idx[:n_train], adjs, adj_Ts, cents, ns)
va_a, va_aT, va_c, va_n = get_split(
    idx[n_train:n_train+n_val], adjs, adj_Ts, cents, ns)
te_a, te_aT, te_c, te_n = get_split(
    idx[n_train+n_val:], adjs, adj_Ts, cents, ns)

print(f"  Train: {len(tr_a)} | "
      f"Val: {len(va_a)} | "
      f"Test: {len(te_a)}\n")

# ═════════════════════════════════════════════════════════════════════
# BUILD MODEL
# ═════════════════════════════════════════════════════════════════════
print("Building model...")
net      = GNN_Bet(ninput=MODEL_SIZE, nhid=HIDDEN,
                   dropout=DROPOUT, learning_rate=LR,
                   weight_decay=WD)
model, _ = net.model_to_device(net)
model    = model.to(device)
optimizer= net.get_optimizer(model)
params   = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {params:,}")
print(f"  Device     : {next(model.parameters()).device}\n")

# ═════════════════════════════════════════════════════════════════════
# TRAINING AND EVALUATION
# ═════════════════════════════════════════════════════════════════════

def train_epoch(adj_list, adj_T_list, cent_list, node_list):
    model.train()
    total = 0.0
    count = 0
    perm  = np.random.permutation(len(adj_list))

    for start in range(0, len(adj_list), BATCH_SIZE):
        batch = perm[start:start+BATCH_SIZE]
        optimizer.zero_grad()
        bloss = torch.tensor(
                    0.0, device=device, requires_grad=True)
        cnt   = 0
        for i in batch:
            a  = sparse_mx_to_torch_sparse_tensor(
                     adj_list[i],   device)
            aT = sparse_mx_to_torch_sparse_tensor(
                     adj_T_list[i], device)
            y  = model(a, aT)
            if torch.isnan(y).any():
                continue
            tv   = torch.from_numpy(
                       cent_list[i]).float().to(device)
            # node_list[i] = actual n, MODEL_SIZE = 5000
            # loss_cal uses actual n for sampling pairs
            loss = loss_cal(y, tv, node_list[i],
                            device, MODEL_SIZE)
            bloss = bloss + loss
            cnt  += 1
        if cnt > 0:
            bloss = bloss / cnt
            bloss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()
            total += float(bloss.detach())
            count += 1
        torch.cuda.empty_cache()

    return total / max(1, count)


def evaluate(adj_list, adj_T_list, cent_list, node_list):
    model.eval()
    kts = []
    with torch.no_grad():
        for i in range(len(adj_list)):
            a  = sparse_mx_to_torch_sparse_tensor(
                     adj_list[i],   device)
            aT = sparse_mx_to_torch_sparse_tensor(
                     adj_T_list[i], device)
            y  = model(a, aT)
            if torch.isnan(y).any():
                continue
            tv = torch.from_numpy(
                     cent_list[i]).float().to(device)
            kt = ranking_correlation(
                     y, tv, node_list[i], MODEL_SIZE)
            if not np.isnan(kt):
                kts.append(kt)
            torch.cuda.empty_cache()

    mean = float(np.mean(kts)) if kts else 0.0
    std  = float(np.std(kts))  if kts else 0.0
    return mean, std

# ═════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════
print("Training...\n")

train_losses = []
val_kts      = []
val_stds     = []
best_val_kt  = -1.0
t0           = time.time()

for epoch in range(EPOCHS):
    loss          = train_epoch(tr_a, tr_aT, tr_c, tr_n)
    val_kt, val_s = evaluate(va_a, va_aT, va_c, va_n)

    train_losses.append(loss)
    val_kts.append(val_kt)
    val_stds.append(val_s)

    if val_kt > best_val_kt:
        best_val_kt = val_kt
        torch.save(model.state_dict(), MODEL_PATH)

    # Save checkpoint every 10 epochs
    if (epoch+1) % 10 == 0:
        torch.save({
            "epoch"        : epoch+1,
            "model_state"  : model.state_dict(),
            "best_val_kt"  : best_val_kt,
            "train_losses" : train_losses,
            "val_kts"      : val_kts,
        }, f"{CHECKPOINT_DIR}/epoch{epoch+1}.pth")

    elapsed = time.time() - t0
    print(f"  Epoch {epoch+1:3d}/{EPOCHS} | "
          f"Loss: {loss:.4f} | "
          f"Val KT: {val_kt:.4f} +/- {val_s:.4f} | "
          f"Best: {best_val_kt:.4f} | "
          f"Time: {elapsed/60:.0f}min")

print(f"\n  Done in {(time.time()-t0)/3600:.2f} hours")
print(f"  Best model saved\n")

# ═════════════════════════════════════════════════════════════════════
# FINAL TEST EVALUATION
# ═════════════════════════════════════════════════════════════════════
print("Final test evaluation...")
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device))
test_kt, test_std = evaluate(te_a, te_aT, te_c, te_n)
print(f"  Test KT : {test_kt:.4f} +/- {test_std:.4f}\n")

# ═════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(range(1, EPOCHS+1), train_losses,
         color="#1f77b4", linewidth=1.5)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("(a) Training Loss — N=5000")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, EPOCHS+1), val_kts,
         color="#9467bd", linewidth=1.5, label="Val KT")
ax2.fill_between(
    range(1, EPOCHS+1),
    [m-s for m, s in zip(val_kts, val_stds)],
    [m+s for m, s in zip(val_kts, val_stds)],
    alpha=0.2, color="#9467bd")
ax2.axhline(y=best_val_kt, color="green",
            linestyle="--", linewidth=1.0,
            label=f"Best = {best_val_kt:.3f}")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Kendall tau")
ax2.set_title("(b) Validation Kendall tau — N=5000")
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle(
    f"GNN Betweenness N=5000 | Exact Labels | {gpu_name}",
    fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_N5000_training.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════
with open(f"{OUTPUT_DIR}/results_N5000_summary.txt",
          "w") as f:
    f.write("GNN Betweenness — N=5000 Results\n")
    f.write("="*45 + "\n\n")
    f.write(f"N={GRAPH_NODES}, p={GRAPH_P}\n")
    f.write(f"Labels: EXACT betweenness\n")
    f.write(f"Train: {len(tr_a)} | "
            f"Val: {len(va_a)} | "
            f"Test: {len(te_a)}\n\n")
    f.write(f"Best val KT : {best_val_kt:.6f}\n")
    f.write(f"Test KT     : "
            f"{test_kt:.6f} +/- {test_std:.6f}\n")
    f.write(f"GPU: {gpu_name}\n")

print("="*55)
print("  COMPLETE — paste into Chapter 4 Section 4.9")
print("="*55)
print(f"  Test KT : {test_kt:.4f} +/- {test_std:.4f}")
print(f"  Labels  : EXACT")
print(f"  GPU     : {gpu_name}")
print("="*55)
print("\n  Download these files:")
print("  results_N5000/betweenness_model_N5000.pth")
print("  results_N5000/results_N5000_summary.txt")
print("  results_N5000/fig_N5000_training.png")
