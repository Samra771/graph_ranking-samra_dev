"""
experiment4_degree_vs_gnn.py
============================

This experiment provides the key evidence that GNN's advantage
lies in generalization to structured graphs, not in raw ER performance.

Run:
    python experiment4_degree_vs_gnn.py

Requires:
    betweenness_model.pth          (ER-trained betweenness)
    betweenness_model_mixed.pth    (Mixed-trained betweenness)
    closeness_model.pth            (ER-trained closeness)
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

from betweennes_model import GNN_Bet
from closeness_model import GNN_Close
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    ranking_correlation
)

SEED = 55
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ═════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════════════
GRAPH_NODES      = 200
GRAPH_SPARSENESS = 0.15
NUM_TEST_GRAPHS  = 200
HIDDEN_LAYERS    = 40
DROPOUT          = 0.4
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 0.01
MODEL_SIZE       = GRAPH_NODES

OUTPUT_DIR = "./paper_figures_degree_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*65}")
print(f"  EXPERIMENT 4: Degree Centrality vs GNN on Structured Graphs")
print(f"{'='*65}")
print(f"  Test graphs per type : {NUM_TEST_GRAPHS}")
print(f"{'='*65}\n")

# ═════════════════════════════════════════════════════════════════════════════
# GRAPH UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def make_er_graph(n, p):
    return nx.erdos_renyi_graph(n, p, directed=False)

def make_ba_graph(n, p):
    m = max(1, int(p * n / 2))
    return nx.barabasi_albert_graph(n, m)

def make_grp_graph(n, p):
    try:
        g = nx.gaussian_random_partition_graph(n=n, s=20, v=5, p_in=0.3, p_out=0.05)
        return nx.Graph(g)
    except Exception:
        return nx.erdos_renyi_graph(n, p, directed=False)

MAKERS = {"ER": make_er_graph, "BA": make_ba_graph, "GRP": make_grp_graph}

def get_adj(graph, model_size):
    n   = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype("float32")
    if n < model_size:
        adj = sp.block_diag([adj, csr_matrix((model_size-n, model_size-n))])
    return adj.tocsr()

def get_adj_T(graph, model_size):
    return get_adj(graph, model_size).T.tocsr()

def get_adj_mod(graph, model_size):
    n   = graph.number_of_nodes()
    adj = nx.adjacency_matrix(graph, nodelist=list(range(n))).astype("float32")
    degrees = np.array(adj.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1.0
    adj_mod = sp.diags(1.0 / degrees) @ adj
    if n < model_size:
        adj_mod = sp.block_diag([adj_mod, csr_matrix((model_size-n, model_size-n))])
    return adj_mod.tocsr()

def to_array(cent_dict, n):
    arr = np.zeros(n, dtype=np.float32)
    for node, val in cent_dict.items():
        if 0 <= node < n:
            arr[node] = float(val)
    return arr

def kt_score(pred, truth, n):
    kt, _ = kendalltau(pred[:n], truth[:n])
    return float(kt) if not np.isnan(kt) else 0.0

# ═════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═════════════════════════════════════════════════════════════════════════════
print("Loading GNN models...")

def load_bet_model(path):
    net   = GNN_Bet(ninput=MODEL_SIZE, nhid=HIDDEN_LAYERS,
                    dropout=DROPOUT, learning_rate=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY)
    model, _ = net.model_to_device(net)
    model    = model.to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  Loaded: {path}")
    else:
        print(f"  NOT FOUND: {path} — using random weights")
    model.eval()
    return model

def load_close_model(path):
    net   = GNN_Close(
    ninput=MODEL_SIZE, nhid=40,
    dropout=0.2,
    learning_rate=1e-3,
    weight_decay=0.0
)
    model, _ = net.model_to_device(net)
    model    = model.to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"  Loaded: {path}")
    else:
        print(f"  NOT FOUND: {path} — using random weights")
    model.eval()
    return model

bet_er_model    = load_bet_model("./betweenness_model.pth")
bet_mixed_model = load_bet_model("./betweenness_model_mixed.pth")
close_er_model  = load_close_model("./closeness_model.pth")

# ═════════════════════════════════════════════════════════════════════════════
# RUN EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

# Store results:
# results[graph_type][method] = list of KT scores
results_bet   = {t: {} for t in ["ER", "BA", "GRP"]}
results_close = {t: {} for t in ["ER", "BA", "GRP"]}

methods_bet   = ["Degree Centrality", "GNN (ER-trained)", "GNN (Mixed-trained)"]
methods_close = ["Degree Centrality", "GNN (ER-trained)"]

for gtype in ["ER", "BA", "GRP"]:
    print(f"\n{'─'*50}")
    print(f"  Graph type: {gtype}")
    print(f"{'─'*50}")

    for m in methods_bet:
        results_bet[gtype][m] = []
    for m in methods_close:
        results_close[gtype][m] = []

    maker = MAKERS[gtype]

    for i in range(NUM_TEST_GRAPHS):
        graph = maker(GRAPH_NODES, GRAPH_SPARSENESS)
        n     = graph.number_of_nodes()

        # Ground truth
        true_bet   = to_array(nx.betweenness_centrality(graph, normalized=True), n)
        true_close = to_array(nx.closeness_centrality(graph), n)

        # Degree centrality
        deg_arr = to_array(nx.degree_centrality(graph), n)
        results_bet[gtype]["Degree Centrality"].append(
            kt_score(deg_arr, true_bet, n))
        results_close[gtype]["Degree Centrality"].append(
            kt_score(deg_arr, true_close, n))

        # GNN predictions
        adj   = get_adj(graph,     MODEL_SIZE)
        adj_T = get_adj_T(graph,   MODEL_SIZE)
        adj_m = get_adj_mod(graph, MODEL_SIZE)

        adj_t  = sparse_mx_to_torch_sparse_tensor(adj,   device)
        adjT_t = sparse_mx_to_torch_sparse_tensor(adj_T, device)
        adjm_t = sparse_mx_to_torch_sparse_tensor(adj_m, device)

        true_bet_t   = torch.from_numpy(true_bet).float().to(device)
        true_close_t = torch.from_numpy(true_close).float().to(device)

        with torch.no_grad():
            # Betweenness ER model
            y = bet_er_model(adj_t, adjT_t)
            if not torch.isnan(y).any():
                results_bet[gtype]["GNN (ER-trained)"].append(
                    ranking_correlation(y, true_bet_t, n, MODEL_SIZE))

            # Betweenness Mixed model
            y = bet_mixed_model(adj_t, adjT_t)
            if not torch.isnan(y).any():
                results_bet[gtype]["GNN (Mixed-trained)"].append(
                    ranking_correlation(y, true_bet_t, n, MODEL_SIZE))

            # Closeness ER model
            y = close_er_model(adj_t, adjm_t)
            if not torch.isnan(y).any():
                results_close[gtype]["GNN (ER-trained)"].append(
                    ranking_correlation(y, true_close_t, n, MODEL_SIZE))

        if (i+1) % 50 == 0:
            print(f"  {i+1}/{NUM_TEST_GRAPHS} graphs processed")

    # Print results for this graph type
    print(f"\n  BETWEENNESS results on {gtype}:")
    for m in methods_bet:
        arr = np.array(results_bet[gtype][m])
        print(f"    {m:<25}: tau = {arr.mean():.4f} +/- {arr.std():.4f}")

    print(f"\n  CLOSENESS results on {gtype}:")
    for m in methods_close:
        arr = np.array(results_close[gtype][m])
        print(f"    {m:<25}: tau = {arr.mean():.4f} +/- {arr.std():.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE CSV
# ═════════════════════════════════════════════════════════════════════════════
with open(f"{OUTPUT_DIR}/degree_vs_gnn_results.csv", "w", encoding="utf-8") as f:
    f.write("Graph Type,Method,Centrality,Mean KT,Std KT\n")
    for gtype in ["ER", "BA", "GRP"]:
        for m in methods_bet:
            arr = np.array(results_bet[gtype][m])
            f.write(f"{gtype},{m},Betweenness,{arr.mean():.6f},{arr.std():.6f}\n")
        for m in methods_close:
            arr = np.array(results_close[gtype][m])
            f.write(f"{gtype},{m},Closeness,{arr.mean():.6f},{arr.std():.6f}\n")
print(f"\n  Saved: {OUTPUT_DIR}/degree_vs_gnn_results.csv")

# ═════════════════════════════════════════════════════════════════════════════
# GENERATE FIGURES
# ═════════════════════════════════════════════════════════════════════════════
STYLE = {
    "figure.dpi": 300, "font.family": "serif",
    "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 10,
    "axes.grid": True, "grid.alpha": 0.3,
}
plt.rcParams.update(STYLE)

graph_types = ["ER", "BA", "GRP"]
x = np.arange(len(graph_types))
width = 0.25

colors = {
    "Degree Centrality"  : "#ff7f0e",
    "GNN (ER-trained)"   : "#6baed6",
    "GNN (Mixed-trained)": "#2ca02c",
}

# ── Figure 1: Betweenness comparison across graph types ───────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for idx, method in enumerate(methods_bet):
    means = [np.mean(results_bet[gt][method]) for gt in graph_types]
    stds  = [np.std(results_bet[gt][method])  for gt in graph_types]
    offset = (idx - 1) * width
    bars = ax.bar(x + offset, means, width, yerr=stds,
                  label=method, color=colors[method],
                  capsize=4, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Kendall tau rank correlation")
ax.set_title("Betweenness Centrality: Degree vs GNN Across Graph Types")
ax.set_xticks(x)
ax.set_xticklabels(["Erdos-Renyi", "Barabasi-Albert", "Gaussian Random Partition"])
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right")
fig.tight_layout()
path = f"{OUTPUT_DIR}/fig_betweenness_degree_vs_gnn.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_betweenness_degree_vs_gnn.png")

# ── Figure 2: Closeness comparison across graph types ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for idx, method in enumerate(methods_close):
    means = [np.mean(results_close[gt][method]) for gt in graph_types]
    stds  = [np.std(results_close[gt][method])  for gt in graph_types]
    offset = (idx - 0.5) * width
    bars = ax.bar(x + offset, means, width, yerr=stds,
                  label=method, color=colors[method],
                  capsize=4, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Kendall tau rank correlation")
ax.set_title("Closeness Centrality: Degree vs GNN Across Graph Types")
ax.set_xticks(x)
ax.set_xticklabels(["Erdos-Renyi", "Barabasi-Albert", "Gaussian Random Partition"])
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right")
fig.tight_layout()
path = f"{OUTPUT_DIR}/fig_closeness_degree_vs_gnn.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_closeness_degree_vs_gnn.png")

# ── Figure 3: Combined side-by-side ───────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for idx, method in enumerate(methods_bet):
    means  = [np.mean(results_bet[gt][method]) for gt in graph_types]
    stds   = [np.std(results_bet[gt][method])  for gt in graph_types]
    offset = (idx - 1) * width
    bars   = ax1.bar(x + offset, means, width, yerr=stds,
                     label=method, color=colors[method],
                     capsize=4, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax1.set_ylabel("Kendall tau")
ax1.set_title("(a) Betweenness Centrality")
ax1.set_xticks(x)
ax1.set_xticklabels(["ER", "BA", "GRP"])
ax1.set_ylim(0, 1.1)
ax1.legend(loc="lower left", fontsize=9)

for idx, method in enumerate(methods_close):
    means  = [np.mean(results_close[gt][method]) for gt in graph_types]
    stds   = [np.std(results_close[gt][method])  for gt in graph_types]
    offset = (idx - 0.5) * width
    bars   = ax2.bar(x + offset, means, width, yerr=stds,
                     label=method, color=colors[method],
                     capsize=4, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax2.set_ylabel("Kendall tau")
ax2.set_title("(b) Closeness Centrality")
ax2.set_xticks(x)
ax2.set_xticklabels(["ER", "BA", "GRP"])
ax2.set_ylim(0, 1.1)
ax2.legend(loc="lower left", fontsize=9)

fig.suptitle("Degree Centrality vs GNN Across Graph Types", fontsize=13)
fig.tight_layout()
path = f"{OUTPUT_DIR}/fig_degree_vs_gnn_combined.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: fig_degree_vs_gnn_combined.png")

# ═════════════════════════════════════════════════════════════════════════════
# PRINT TABLE AND KEY FINDINGS
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"   COMPARISON TABLE")
print(f"{'='*65}")
print(f"\n  Betweenness Centrality:")
print(f"  {'Method':<25} {'ER':>12} {'BA':>12} {'GRP':>12}")
print(f"  {'-'*60}")
for m in methods_bet:
    row = f"  {m:<25}"
    for gt in ["ER", "BA", "GRP"]:
        arr  = np.array(results_bet[gt][m])
        row += f"  {arr.mean():.3f}+/-{arr.std():.3f}"
    print(row)

print(f"\n  Closeness Centrality:")
print(f"  {'Method':<25} {'ER':>12} {'BA':>12} {'GRP':>12}")
print(f"  {'-'*60}")
for m in methods_close:
    row = f"  {m:<25}"
    for gt in ["ER", "BA", "GRP"]:
        arr  = np.array(results_close[gt][m])
        row += f"  {arr.mean():.3f}+/-{arr.std():.3f}"
    print(row)

print(f"\n{'='*65}")
print(f"  KEY FINDINGS FOR PAPER")
print(f"{'='*65}")

deg_er   = np.mean(results_bet["ER"]["Degree Centrality"])
deg_ba   = np.mean(results_bet["BA"]["Degree Centrality"])
deg_grp  = np.mean(results_bet["GRP"]["Degree Centrality"])
gnn_ba   = np.mean(results_bet["BA"]["GNN (Mixed-trained)"])
gnn_grp  = np.mean(results_bet["GRP"]["GNN (Mixed-trained)"])
gnn_er_b = np.mean(results_bet["ER"]["GNN (ER-trained)"])

print(f"  All figures saved to: {OUTPUT_DIR}/")
print(f"    fig_degree_vs_gnn_combined.png  -> Figure 4.5 (main comparison)")
print(f"    degree_vs_gnn_results.csv       -> Table 3\n")
