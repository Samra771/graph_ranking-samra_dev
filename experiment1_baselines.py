"""
experiment1_baselines.py
========================
Compares GNN betweenness and closeness centrality approximations against:

Baseline 1 — Degree Centrality (trivial baseline)
  Computes in one pass: O(N). If degree correlates well with betweenness/
  closeness, it serves as a strong trivial baseline. Your GNN must beat this.

Baseline 2 — Brandes Approximation (standard practical baseline)
  The standard sampling-based approximation for betweenness centrality.
  Samples k pivot nodes and estimates betweenness from shortest path trees.
  NetworkX implements this as betweenness_centrality(G, k=k).
  Complexity: O(k * N * E) instead of O(N * E).
  We test k = 10, 20, 50 pivots.

Baseline 3 — Random Ranking (lower bound)
  Assigns random scores to nodes. Gives expected tau = 0.
  Confirms that your evaluation metric is working correctly.

Output:
  - Comparison table (saved as CSV and printed)
  - Bar chart comparing all methods
  - Saved to ./paper_figures_baselines/

Run:
  python experiment1_baselines.py

Note: This script loads yourTRAINED models from:
  ./betweenness_model.pth
  ./closeness_model.pth
Make sure you have run run_experiment.py and run_experiment_closeness.py
first so these model files exist.
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
import pandas as pd
import random
import time

from betweennes_model import GNN_Bet
from closeness_model import GNN_Close
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    ranking_correlation
)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 123   # Different seed from training — ensures fair test
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
GRAPH_NODES      = 200
GRAPH_SPARSENESS = 0.15
NUM_TEST_GRAPHS  = 200    # Enough for statistical significance

HIDDEN_LAYERS    = 40
DROPOUT          = 0.4
LEARNING_RATE    = 1e-3   # Betweenness
WEIGHT_DECAY     = 0.01
MODEL_SIZE       = GRAPH_NODES

BRANDES_K_VALUES = [10, 20, 50]   # Number of pivot nodes for Brandes approx

OUTPUT_DIR       = "./paper_figures_baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*65}")
print(f"  EXPERIMENT 1: Baseline Comparison")
print(f"{'='*65}")
print(f"  Device         : {device}")
print(f"  Test graphs    : {NUM_TEST_GRAPHS}")
print(f"  Graph type     : Undirected ER (N={GRAPH_NODES}, p={GRAPH_SPARSENESS})")
print(f"{'='*65}\n")

# ═════════════════════════════════════════════════════════════════════════════
# LOAD TRAINED GNN MODELS
# ═════════════════════════════════════════════════════════════════════════════
print("Loading trained GNN models...")

# Betweenness GNN
bet_network = GNN_Bet(
    ninput=MODEL_SIZE, nhid=HIDDEN_LAYERS,
    dropout=DROPOUT, learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
bet_model, _ = bet_network.model_to_device(bet_network)
bet_model     = bet_model.to(device)

if os.path.exists("./betweenness_model.pth"):
    bet_model.load_state_dict(torch.load("./betweenness_model.pth", map_location=device))
    bet_model.eval()
    print("  Betweenness model loaded.")
else:
    print("  WARNING: betweenness_model.pth not found.")
    print("  Run run_experiment.py first to train and save the model.")
    bet_model = None

# Closeness GNN
# Find the closeness model creation line and change to:
close_network = GNN_Close(
    ninput=MODEL_SIZE, nhid=40,
    dropout=0.2,          # closeness best config
    learning_rate=1e-3,
    weight_decay=0.0      # closeness best config
)
close_model, _ = close_network.model_to_device(close_network)
close_model     = close_model.to(device)

if os.path.exists("./closeness_model.pth"):
    close_model.load_state_dict(torch.load("./closeness_model.pth", map_location=device))
    close_model.eval()
    print("  Closeness model loaded.")
else:
    print("  WARNING: closeness_model.pth not found.")
    print("  Run run_experiment_closeness_fixed.py first.")
    close_model = None

# ═════════════════════════════════════════════════════════════════════════════
# GRAPH UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def make_er_graph(n, p):
    """Undirected ER graph — same type used for closeness training."""
    return nx.erdos_renyi_graph(n, p, directed=False)


def get_adj(graph, model_size):
    """Standard adjacency matrix padded to model_size."""
    n   = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype(np.float32)
    if n < model_size:
        adj = sp.block_diag([adj, csr_matrix((model_size-n, model_size-n))])
    return adj.tocsr()


def get_adj_mod(graph, model_size):
    """Degree-normalised adjacency for closeness GNN."""
    n   = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype(np.float32)
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    D_inv   = sp.diags(1.0 / degrees)
    adj_mod = D_inv @ adj
    if n < model_size:
        adj_mod = sp.block_diag([adj_mod, csr_matrix((model_size-n, model_size-n))])
    return adj_mod.tocsr()


def get_adj_T(graph, model_size):
    """Transpose adjacency matrix for betweenness GNN."""
    adj = get_adj(graph, model_size)
    return adj.T.tocsr()


def centrality_to_array(cent_dict, n):
    arr = np.zeros(n, dtype=np.float32)
    for node, val in cent_dict.items():
        if 0 <= node < n:
            arr[node] = float(val)
    return arr


def kendall_tau(pred, truth, n):
    """Compute Kendall tau between first n elements of pred and truth."""
    kt, _ = kendalltau(pred[:n], truth[:n])
    return float(kt) if not np.isnan(kt) else 0.0

# ═════════════════════════════════════════════════════════════════════════════
# GENERATE TEST GRAPHS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nGenerating {NUM_TEST_GRAPHS} test graphs...")
graphs = []
for i in range(NUM_TEST_GRAPHS):
    graphs.append(make_er_graph(GRAPH_NODES, GRAPH_SPARSENESS))
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{NUM_TEST_GRAPHS}")

# ═════════════════════════════════════════════════════════════════════════════
# RUN ALL METHODS
# ═════════════════════════════════════════════════════════════════════════════

# Storage for results
results = {
    "Random Ranking"          : {"bet": [], "close": [], "time_bet": [], "time_close": []},
    "Degree Centrality"       : {"bet": [], "close": [], "time_bet": [], "time_close": []},
    "GNN"                     : {"bet": [], "close": [], "time_bet": [], "time_close": []},
}
for k in BRANDES_K_VALUES:
    results[f"Brandes k={k}"] = {"bet": [], "close": [], "time_bet": [], "time_close": []}

print(f"\nRunning all methods on {NUM_TEST_GRAPHS} test graphs...")
print(f"{'Graph':>6} ", end="", flush=True)

for idx, graph in enumerate(graphs):
    n = graph.number_of_nodes()

    # ── Ground truth ──────────────────────────────────────────────────────────
    true_bet   = centrality_to_array(nx.betweenness_centrality(graph, normalized=True), n)
    true_close = centrality_to_array(nx.closeness_centrality(graph), n)

    # ── Baseline 1: Random Ranking ────────────────────────────────────────────
    random_scores = np.random.rand(n)
    results["Random Ranking"]["bet"].append(kendall_tau(random_scores, true_bet, n))
    results["Random Ranking"]["close"].append(kendall_tau(random_scores, true_close, n))
    results["Random Ranking"]["time_bet"].append(0.0)
    results["Random Ranking"]["time_close"].append(0.0)

    # ── Baseline 2: Degree Centrality ─────────────────────────────────────────
    t0 = time.perf_counter()
    deg_cent = nx.degree_centrality(graph)
    t_deg    = time.perf_counter() - t0
    deg_arr  = centrality_to_array(deg_cent, n)

    results["Degree Centrality"]["bet"].append(kendall_tau(deg_arr, true_bet, n))
    results["Degree Centrality"]["close"].append(kendall_tau(deg_arr, true_close, n))
    results["Degree Centrality"]["time_bet"].append(t_deg)
    results["Degree Centrality"]["time_close"].append(t_deg)

    # ── Baseline 3: Brandes Approximation (betweenness only) ─────────────────
    for k in BRANDES_K_VALUES:
        t0 = time.perf_counter()
        brandes_bc  = nx.betweenness_centrality(graph, k=k, normalized=True)
        t_brandes   = time.perf_counter() - t0
        brandes_arr = centrality_to_array(brandes_bc, n)
        kt_brandes  = kendall_tau(brandes_arr, true_bet, n)
        results[f"Brandes k={k}"]["bet"].append(kt_brandes)
        results[f"Brandes k={k}"]["close"].append(np.nan)  # N/A for closeness
        results[f"Brandes k={k}"]["time_bet"].append(t_brandes)
        results[f"Brandes k={k}"]["time_close"].append(np.nan)

    # ── GNN Predictions ───────────────────────────────────────────────────────
    adj   = get_adj(graph, MODEL_SIZE)
    adj_T = get_adj_T(graph, MODEL_SIZE)
    adj_m = get_adj_mod(graph, MODEL_SIZE)

    adj_t   = sparse_mx_to_torch_sparse_tensor(adj,   device)
    adj_T_t = sparse_mx_to_torch_sparse_tensor(adj_T, device)
    adj_m_t = sparse_mx_to_torch_sparse_tensor(adj_m, device)

    # GNN Betweenness
    if bet_model is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            y_bet = bet_model(adj_t, adj_T_t).cpu().numpy().flatten()[:n]
        t_gnn_bet = time.perf_counter() - t0
        results["GNN"]["bet"].append(kendall_tau(y_bet, true_bet, n))
        results["GNN"]["time_bet"].append(t_gnn_bet)
    else:
        results["GNN"]["bet"].append(np.nan)
        results["GNN"]["time_bet"].append(np.nan)

    # GNN Closeness
    if close_model is not None:
        t0 = time.perf_counter()
        with torch.no_grad():
            y_close = close_model(adj_t, adj_m_t).cpu().numpy().flatten()[:n]
        t_gnn_close = time.perf_counter() - t0
        results["GNN"]["close"].append(kendall_tau(y_close, true_close, n))
        results["GNN"]["time_close"].append(t_gnn_close)
    else:
        results["GNN"]["close"].append(np.nan)
        results["GNN"]["time_close"].append(np.nan)

    if (idx + 1) % 20 == 0:
        print(f"\r  Completed {idx+1}/{NUM_TEST_GRAPHS} graphs", end="", flush=True)

print(f"\r  Completed {NUM_TEST_GRAPHS}/{NUM_TEST_GRAPHS} graphs\n")

# ═════════════════════════════════════════════════════════════════════════════
# COMPUTE SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  RESULTS SUMMARY")
print(f"{'='*65}")

rows = []
for method, data in results.items():
    bet_arr   = np.array([x for x in data["bet"]   if not np.isnan(x)])
    close_arr = np.array([x for x in data["close"] if not np.isnan(x)])
    t_bet_arr = np.array([x for x in data["time_bet"]   if not np.isnan(x)])
    t_cl_arr  = np.array([x for x in data["time_close"] if not np.isnan(x)])

    row = {
        "Method"              : method,
        "KT Betweenness"      : f"{bet_arr.mean():.4f}" if len(bet_arr) > 0 else "N/A",
        "KT Bet Std"          : f"{bet_arr.std():.4f}"  if len(bet_arr) > 0 else "N/A",
        "KT Closeness"        : f"{close_arr.mean():.4f}" if len(close_arr) > 0 else "N/A",
        "KT Close Std"        : f"{close_arr.std():.4f}"  if len(close_arr) > 0 else "N/A",
        "Time Bet (ms)"       : f"{t_bet_arr.mean()*1000:.2f}" if len(t_bet_arr) > 0 else "N/A",
        "Time Close (ms)"     : f"{t_cl_arr.mean()*1000:.2f}"  if len(t_cl_arr)  > 0 else "N/A",
    }
    rows.append(row)
    print(f"\n  Method: {method}")
    print(f"    Betweenness KT : {row['KT Betweenness']} ± {row['KT Bet Std']}")
    print(f"    Closeness KT   : {row['KT Closeness']} ± {row['KT Close Std']}")
    print(f"    Inference time : Bet={row['Time Bet (ms)']}ms | Close={row['Time Close (ms)']}ms")

# ── Save as CSV for paper table ───────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = f"{OUTPUT_DIR}/baseline_comparison.csv"
df.to_csv(csv_path, index=False)
print(f"\n  Results saved to: {csv_path}")

# ═════════════════════════════════════════════════════════════════════════════
# GENERATE FIGURES
# ═════════════════════════════════════════════════════════════════════════════
STYLE = {
    "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 10,
}
plt.rcParams.update(STYLE)

# ── Figure 1: Betweenness comparison bar chart ────────────────────────────────
methods_bet  = []
means_bet    = []
stds_bet     = []
colors_bet   = []

color_map = {
    "Random Ranking"   : "#cccccc",
    "Degree Centrality": "#ff7f0e",
    "Brandes k=10"     : "#aec7e8",
    "Brandes k=20"     : "#6baed6",
    "Brandes k=50"     : "#2171b5",
    "GNN"              : "#2ca02c",
}

for method, data in results.items():
    arr = np.array([x for x in data["bet"] if not np.isnan(x)])
    if len(arr) > 0:
        methods_bet.append(method)
        means_bet.append(arr.mean())
        stds_bet.append(arr.std())
        colors_bet.append(color_map.get(method, "#888888"))

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(methods_bet, means_bet, yerr=stds_bet,
              color=colors_bet, capsize=5, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Kendall τ Rank Correlation")
ax.set_title("Betweenness Centrality Approximation: Method Comparison")
ax.set_ylim(0, 1.05)
ax.axhline(y=means_bet[-1], color="#2ca02c", linestyle="--",
           linewidth=1.0, alpha=0.5, label="GNN performance")
ax.legend()
plt.xticks(rotation=15, ha="right")

# Annotate bars with values
for bar, mean in zip(bars, means_bet):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_betweenness_comparison_methods.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_betweenness_comparison_methods.png")

# ── Figure 2: Closeness comparison bar chart ──────────────────────────────────
methods_cl = []
means_cl   = []
stds_cl    = []
colors_cl  = []

for method, data in results.items():
    arr = np.array([x for x in data["close"] if not np.isnan(x)])
    if len(arr) > 0:
        methods_cl.append(method)
        means_cl.append(arr.mean())
        stds_cl.append(arr.std())
        colors_cl.append(color_map.get(method, "#888888"))

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(methods_cl, means_cl, yerr=stds_cl,
              color=colors_cl, capsize=5, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Kendall τ Rank Correlation")
ax.set_title("Closeness Centrality Approximation: Method Comparison")
ax.set_ylim(0, 1.05)
for bar, mean in zip(bars, means_cl):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=10, ha="right")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_closeness_comparison_methods.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_closeness_comparison_methods.png")

# ── Figure 3: Brandes k vs Kendall tau (betweenness only) ────────────────────
k_vals  = BRANDES_K_VALUES
kt_vals = []
kt_stds_b = []
for k in k_vals:
    arr = np.array(results[f"Brandes k={k}"]["bet"])
    kt_vals.append(arr.mean())
    kt_stds_b.append(arr.std())

gnn_kt = np.array([x for x in results["GNN"]["bet"] if not np.isnan(x)]).mean()

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(k_vals, kt_vals, yerr=kt_stds_b,
            marker="o", color="#2171b5", linewidth=1.5,
            capsize=4, label="Brandes approximation")
ax.axhline(y=gnn_kt, color="#2ca02c", linestyle="--",
           linewidth=1.5, label=f"GNN (τ = {gnn_kt:.3f})")
ax.set_xlabel("Number of pivot nodes k")
ax.set_ylabel("Kendall τ")
ax.set_title("Brandes Approximation: Accuracy vs Number of Pivots")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fig_brandes_vs_gnn.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_brandes_vs_gnn.png")

print(f"\n  All baseline figures saved to: {OUTPUT_DIR}/")
