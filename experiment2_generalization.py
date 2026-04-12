"""
experiment2_generalization.py
==============================
Generalization experiment to test whether GNNs trained on Erdos-Renyi graphs can generalize to:
  - Barabasi-Albert (scale-free) graphs
  - Gaussian Random Partition graphs (community structure)
  - Mixed graph types

Tests whether GNNs trained on Erdos-Renyi graphs generalize to:
  - Barabasi-Albert (scale-free) graphs
  - Gaussian Random Partition graphs (community structure)
  - Mixed graph types

This is critical for a point of interest because:
  Real-world networks are NOT Erdos-Renyi. If your GNN only works on
  the graph type it was trained on, the contribution is limited.
  If it generalizes, it is a much stronger result.

Three experimental conditions:
  Condition A: Train ER - Test ER       (in-distribution — your existing result)
  Condition B: Train ER - Test BA       (out-of-distribution generalization)
  Condition C: Train ER - Test GRP      (out-of-distribution generalization)
  Condition D: Train Mixed - Test Mixed (cross-type training)

Output:
  - Generalization table (CSV)
  - Heatmap of tau across train/test type combinations
  - Saved to ./paper_figures_generalization/

Run:
  python experiment2_generalization.py
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
import random
import time

from betweennes_model import GNN_Bet
from closeness_model import GNN_Close
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    loss_cal,
    ranking_correlation
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 99
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
GRAPH_NODES      = 200
GRAPH_SPARSENESS = 0.15
TRAIN_GRAPHS     = 1600
TEST_GRAPHS      = 200
HIDDEN_LAYERS    = 20
EPOCHS_QUICK     = 50     # Fewer epochs for cross-type training (faster)
LEARNING_RATE    = 1e-4
DROPOUT          = 0.2
WEIGHT_DECAY     = 0.01
BATCH_SIZE       = 16
MODEL_SIZE       = GRAPH_NODES

OUTPUT_DIR = "./paper_figures_generalization"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*65}")
print(f"  EXPERIMENT 2: Generalization Across Graph Types")
print(f"{'='*65}")
print(f"  Device      : {device}")
print(f"  Train graphs: {TRAIN_GRAPHS} per type")
print(f"  Test graphs : {TEST_GRAPHS} per type")
print(f"{'='*65}\n")

# ═════════════════════════════════════════════════════════════════════════════
# GRAPH GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def make_er_graph(n, p):
    return nx.erdos_renyi_graph(n, p, directed=False)

def make_ba_graph(n, p):
    """Barabasi-Albert scale-free graph. m = avg connections."""
    m = max(1, int(p * n / 2))
    return nx.barabasi_albert_graph(n, m)

def make_grp_graph(n, p):
    """
    Gaussian Random Partition graph — community structure.
    s = mean community size, v = variance, p_in = intra-community edge prob,
    p_out = inter-community edge prob.
    """
    try:
        g = nx.gaussian_random_partition_graph(
            n=n, s=20, v=5, p_in=0.3, p_out=0.05
        )
        return nx.Graph(g)  # Ensure undirected
    except Exception:
        # Fallback to ER if GRP generation fails
        return nx.erdos_renyi_graph(n, p, directed=False)

GRAPH_MAKERS = {
    "ER"  : make_er_graph,
    "BA"  : make_ba_graph,
    "GRP" : make_grp_graph,
}

def get_adj(graph, model_size):
    n   = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype(np.float32)
    if n < model_size:
        adj = sp.block_diag([adj, csr_matrix((model_size-n, model_size-n))])
    return adj.tocsr()

def get_adj_T(graph, model_size):
    return get_adj(graph, model_size).T.tocsr()

def get_adj_mod(graph, model_size):
    n   = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype(np.float32)
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    D_inv   = sp.diags(1.0 / degrees)
    adj_mod = D_inv @ adj
    if n < model_size:
        adj_mod = sp.block_diag([adj_mod, csr_matrix((model_size-n, model_size-n))])
    return adj_mod.tocsr()

def centrality_to_array(cent_dict, n):
    arr = np.zeros(n, dtype=np.float32)
    for node, val in cent_dict.items():
        if 0 <= node < n:
            arr[node] = float(val)
    return arr

def build_dataset(graph_type, num_graphs, mode="betweenness"):
    """
    Generate a dataset of graphs of a given type.
    mode: 'betweenness' or 'closeness'
    Returns lists of adj, adj2, centrality, node_counts.
    """
    maker = GRAPH_MAKERS[graph_type]
    adjs, adj2s, cents, ns = [], [], [], []

    for i in range(num_graphs):
        g = maker(GRAPH_NODES, GRAPH_SPARSENESS)
        n = g.number_of_nodes()

        adj = get_adj(g, MODEL_SIZE)
        if mode == "betweenness":
            adj2 = get_adj_T(g, MODEL_SIZE)
            cent = nx.betweenness_centrality(g, normalized=True)
        else:
            adj2 = get_adj_mod(g, MODEL_SIZE)
            cent = nx.closeness_centrality(g)

        adjs.append(adj)
        adj2s.append(adj2)
        cents.append(centrality_to_array(cent, n))
        ns.append(n)

        if (i+1) % 400 == 0:
            print(f"    {i+1}/{num_graphs} graphs generated")

    return adjs, adj2s, cents, ns

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION (reusable)
# ═════════════════════════════════════════════════════════════════════════════

def train_model(model, optimizer, adjs, adj2s, cents, ns, epochs, mode="betweenness"):
    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(len(adjs))
        total_loss = 0.0
        for batch_start in range(0, len(adjs), BATCH_SIZE):
            batch_idx = perm[batch_start:batch_start+BATCH_SIZE]
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            count = 0
            for i in batch_idx:
                adj  = sparse_mx_to_torch_sparse_tensor(adjs[i],  device)
                adj2 = sparse_mx_to_torch_sparse_tensor(adj2s[i], device)
                y    = model(adj, adj2)
                if torch.isnan(y).any():
                    continue
                tv   = torch.from_numpy(cents[i]).float().to(device)
                loss = loss_cal(y, tv, ns[i], device, MODEL_SIZE)
                batch_loss = batch_loss + loss
                count += 1
            if count > 0:
                batch_loss = batch_loss / count
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(batch_loss.detach())
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")


def evaluate_model(model, adjs, adj2s, cents, ns):
    """Returns mean and std Kendall tau on a dataset."""
    model.eval()
    kt_scores = []
    with torch.no_grad():
        for i in range(len(adjs)):
            adj  = sparse_mx_to_torch_sparse_tensor(adjs[i],  device)
            adj2 = sparse_mx_to_torch_sparse_tensor(adj2s[i], device)
            y    = model(adj, adj2)
            if torch.isnan(y).any():
                continue
            tv   = torch.from_numpy(cents[i]).float().to(device)
            kt   = ranking_correlation(y, tv, ns[i], MODEL_SIZE)
            if not np.isnan(kt):
                kt_scores.append(kt)
    return np.mean(kt_scores), np.std(kt_scores)

# ═════════════════════════════════════════════════════════════════════════════
# CONDITION A: Load pre-trained ER model, test on all types
# ═════════════════════════════════════════════════════════════════════════════
print("CONDITION A: Pre-trained ER model → test on all graph types")
print("  (Tests out-of-distribution generalization of your existing model)")

results_generalization = {}

# Load pre-trained betweenness model
bet_network = GNN_Bet(ninput=MODEL_SIZE, nhid=HIDDEN_LAYERS, dropout=DROPOUT,
                      learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
bet_model, _ = bet_network.model_to_device(bet_network)
bet_model = bet_model.to(device)

if os.path.exists("./betweenness_model.pth"):
    bet_model.load_state_dict(torch.load("./betweenness_model.pth", map_location=device))
    bet_model.eval()
    print("  Betweenness model loaded from file.")

    for test_type in ["ER", "BA", "GRP"]:
        print(f"  Testing on {test_type} graphs...")
        adjs, adj2s, cents, ns = build_dataset(test_type, TEST_GRAPHS, "betweenness")
        mean_kt, std_kt = evaluate_model(bet_model, adjs, adj2s, cents, ns)
        key = f"ER->{test_type} (Betweenness)"
        results_generalization[key] = (mean_kt, std_kt)
        print(f"    Kendall tau = {mean_kt:.4f} +/- {std_kt:.4f}")
else:
    print("  betweenness_model.pth not found — skipping pre-trained evaluation")

# Load pre-trained closeness model
close_network = GNN_Close(ninput=MODEL_SIZE, nhid=HIDDEN_LAYERS, dropout=DROPOUT,
                          learning_rate=5e-4, weight_decay=WEIGHT_DECAY)
close_model, _ = close_network.model_to_device(close_network)
close_model = close_model.to(device)

if os.path.exists("./closeness_model.pth"):
    close_model.load_state_dict(torch.load("./closeness_model.pth", map_location=device))
    close_model.eval()
    print("\n  Closeness model loaded from file.")

    for test_type in ["ER", "BA", "GRP"]:
        print(f"  Testing on {test_type} graphs (closeness)...")
        adjs, adj2s, cents, ns = build_dataset(test_type, TEST_GRAPHS, "closeness")
        mean_kt, std_kt = evaluate_model(close_model, adjs, adj2s, cents, ns)
        key = f"ER->{test_type} (Closeness)"
        results_generalization[key] = (mean_kt, std_kt)
        print(f"    Kendall tau = {mean_kt:.4f} +/- {std_kt:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# CONDITION D: Train on Mixed types, test on all types
# ══════════════════════════
print("\nCONDITION D: Train on Mixed (ER+BA+GRP) → test on all types")
print("  (Tests whether diversity in training improves generalization)")

# Build mixed training set for betweenness
print("  Building mixed training dataset (betweenness)...")
mixed_adjs, mixed_adj2s, mixed_cents, mixed_ns = [], [], [], []
per_type = TRAIN_GRAPHS // 3
for gtype in ["ER", "BA", "GRP"]:
    print(f"    Generating {per_type} {gtype} graphs...")
    a, a2, c, n = build_dataset(gtype, per_type, "betweenness")
    mixed_adjs  += a
    mixed_adj2s += a2
    mixed_cents += c
    mixed_ns    += n

# Shuffle mixed dataset
perm = np.random.permutation(len(mixed_adjs))
mixed_adjs  = [mixed_adjs[i]  for i in perm]
mixed_adj2s = [mixed_adj2s[i] for i in perm]
mixed_cents = [mixed_cents[i] for i in perm]
mixed_ns    = [mixed_ns[i]    for i in perm]

# Train mixed betweenness model
print(f"\n  Training mixed betweenness model ({EPOCHS_QUICK} epochs)...")
mixed_bet = GNN_Bet(ninput=MODEL_SIZE, nhid=HIDDEN_LAYERS, dropout=DROPOUT,
                    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
mixed_bet, _ = mixed_bet.model_to_device(mixed_bet)
mixed_bet    = mixed_bet.to(device)
mixed_bet_opt = mixed_bet.get_optimizer(mixed_bet)
train_model(mixed_bet, mixed_bet_opt, mixed_adjs, mixed_adj2s,
            mixed_cents, mixed_ns, EPOCHS_QUICK, "betweenness")
torch.save(mixed_bet.state_dict(), "./betweenness_model_mixed.pth")

# Test mixed model on all graph types
mixed_bet.eval()
for test_type in ["ER", "BA", "GRP"]:
    print(f"  Testing mixed model on {test_type}...")
    adjs, adj2s, cents, ns = build_dataset(test_type, TEST_GRAPHS, "betweenness")
    mean_kt, std_kt = evaluate_model(mixed_bet, adjs, adj2s, cents, ns)
    key = f"Mixed->{test_type} (Betweenness)"
    results_generalization[key] = (mean_kt, std_kt)
    print(f"    Kendall tau = {mean_kt:.4f} +/- {std_kt:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# PRINT AND SAVE RESULTS
# ══════════════════════
print(f"  GENERALIZATION RESULTS")
print(f"{'='*65}")
for key, (mean, std) in results_generalization.items():
    print(f"  {key:<40} tau = {mean:.4f} +/- {std:.4f}")

# Save CSV
with open(f"{OUTPUT_DIR}/generalization_results.csv", "w", encoding="utf-8") as f:
    f.write("Condition,Mean KT,Std KT\n")
    for key, (mean, std) in results_generalization.items():
        f.write(f"{key},{mean:.6f},{std:.6f}\n")
print(f"\n  Saved: {OUTPUT_DIR}/generalization_results.csv")

# ── Figure: Generalization heatmap ────────────────────────────────────────────
STYLE = {"figure.dpi": 300, "font.family": "serif", "font.size": 11}
plt.rcParams.update(STYLE)

# Build matrix for betweenness heatmap
train_types = ["ER", "Mixed"]
test_types  = ["ER", "BA", "GRP"]
bet_matrix  = np.zeros((len(train_types), len(test_types)))
for ti, tt in enumerate(train_types):
    for tj, te in enumerate(test_types):
        key = f"{tt}->{te} (Betweenness)"
        if key in results_generalization:
            bet_matrix[ti, tj] = results_generalization[key][0]

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(bet_matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Kendall tau")
ax.set_xticks(range(len(test_types)));  ax.set_xticklabels(test_types)
ax.set_yticks(range(len(train_types))); ax.set_yticklabels(train_types)
ax.set_xlabel("Test Graph Type")
ax.set_ylabel("Training Graph Type")
ax.set_title("Betweenness Centrality: Generalization Across Graph Types")

for i in range(len(train_types)):
    for j in range(len(test_types)):
        ax.text(j, i, f"{bet_matrix[i,j]:.3f}",
                ha="center", va="center", fontsize=12,
                color="black" if bet_matrix[i,j] > 0.4 else "white")

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_generalization_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_generalization_heatmap.png")
print(f"\n  All generalization figures saved to: {OUTPUT_DIR}/\n")
