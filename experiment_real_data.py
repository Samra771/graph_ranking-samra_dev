"""
experiment_real_data.py
========================
Zero-shot validation of trained GNN models on real-world networks.

Betweenness model : betweenness_model_N5000.pth (N=5000)
Closeness model   : closeness_model.pth (N=200)

Datasets required in repo folder:
  1. C-elegans-frontal.txt.gz
     https://snap.stanford.edu/data/C-elegans-frontal.txt.gz

  2. email-Eu-core.txt.gz
     https://snap.stanford.edu/data/email-Eu-core.txt.gz

  3. power.gml  (unzip power.zip)
     http://www-personal.umich.edu/~mejn/netdata/power.zip

Run:
    python experiment_real_data.py

Output:
    paper_figures_real_data/real_world_results.txt
    paper_figures_real_data/fig_real_world_validation.png
"""

import os
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import gzip
import shutil

from betweennes_model import GNN_Bet
from closeness_model import GNN_Close
from betweennes_training_library import (
    sparse_mx_to_torch_sparse_tensor,
    ranking_correlation
)

# ═════════════════════════════════════════════════════════════════════
# SETTINGS
# ═════════════════════════════════════════════════════════════════════

# Betweenness model — trained at N=5000
BET_MODEL_SIZE = 5000
HIDDEN_BET     = 40
DROPOUT_BET    = 0.4
LR_BET         = 1e-3
WD_BET         = 0.01
BET_MODEL_FILE = "./betweenness_model_N5000.pth"

# Closeness model — trained at N=200
CLOSE_MODEL_SIZE = 200
HIDDEN_CLOSE     = 40
DROPOUT_CLOSE    = 0.2
LR_CLOSE         = 1e-3
WD_CLOSE         = 0.0
CLOSE_MODEL_FILE = "./closeness_model.pth"

OUTPUT_DIR = "./paper_figures_real_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available()
                       else "cpu")

print(f"\n{'='*60}")
print(f"  Real-World Network Validation")
print(f"{'='*60}")
print(f"  Device            : {device}")
print(f"  Betweenness model : N={BET_MODEL_SIZE}")
print(f"  Closeness model   : N={CLOSE_MODEL_SIZE}")
print(f"  Mode              : Zero-shot (no retraining)")
print(f"{'='*60}\n")

# ═════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═════════════════════════════════════════════════════════════════════

def load_bet_model():
    net      = GNN_Bet(ninput=BET_MODEL_SIZE, nhid=HIDDEN_BET,
                       dropout=DROPOUT_BET, learning_rate=LR_BET,
                       weight_decay=WD_BET)
    model, _ = net.model_to_device(net)
    model    = model.to(device)
    if os.path.exists(BET_MODEL_FILE):
        model.load_state_dict(
            torch.load(BET_MODEL_FILE, map_location=device))
        print(f"  Betweenness model loaded "
              f"({BET_MODEL_FILE}).")
    else:
        print(f"  ERROR: {BET_MODEL_FILE} not found.")
        return None
    model.eval()
    return model

def load_close_model():
    net      = GNN_Close(ninput=CLOSE_MODEL_SIZE,
                         nhid=HIDDEN_CLOSE,
                         dropout=DROPOUT_CLOSE,
                         learning_rate=LR_CLOSE,
                         weight_decay=WD_CLOSE)
    model, _ = net.model_to_device(net)
    model    = model.to(device)
    if os.path.exists(CLOSE_MODEL_FILE):
        model.load_state_dict(
            torch.load(CLOSE_MODEL_FILE, map_location=device))
        print(f"  Closeness model loaded "
              f"({CLOSE_MODEL_FILE}).")
    else:
        print(f"  ERROR: {CLOSE_MODEL_FILE} not found.")
        return None
    model.eval()
    return model

print("Loading trained GNN models...")
bet_model   = load_bet_model()
close_model = load_close_model()

# ═════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═════════════════════════════════════════════════════════════════════

def unzip_if_needed(gz_path, out_path):
    if not os.path.exists(out_path) and os.path.exists(gz_path):
        print(f"  Unzipping {gz_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def load_celegans():
    gz_path  = "./C-elegans-frontal.txt.gz"
    txt_path = "./celegans.txt"
    for path in [txt_path, "./C-elegans-frontal.txt"]:
        if os.path.exists(path):
            txt_path = path
            break
    else:
        unzip_if_needed(gz_path, txt_path)
    if not os.path.exists(txt_path):
        print("  C. Elegans not found.")
        print("  Download: https://snap.stanford.edu/data/"
              "C-elegans-frontal.txt.gz")
        return None, "C. Elegans Neural"
    g = nx.read_edgelist(txt_path, comments='#',
                         create_using=nx.Graph())
    g = nx.convert_node_labels_to_integers(g)
    gcc = max(nx.connected_components(g), key=len)
    g   = g.subgraph(gcc).copy()
    g   = nx.convert_node_labels_to_integers(g)
    print(f"  C. Elegans     : {g.number_of_nodes()} nodes, "
          f"{g.number_of_edges()} edges")
    return g, "C. Elegans Neural"

def load_email_eucore():
    gz_path  = "./email-Eu-core.txt.gz"
    txt_path = "./email-Eu-core.txt"
    unzip_if_needed(gz_path, txt_path)
    if not os.path.exists(txt_path):
        print("  Email-Eu-Core not found.")
        print("  Download: https://snap.stanford.edu/data/"
              "email-Eu-core.txt.gz")
        return None, "Email-Eu-Core"
    g = nx.read_edgelist(txt_path, comments='#',
                         create_using=nx.Graph())
    g = nx.convert_node_labels_to_integers(g)
    gcc = max(nx.connected_components(g), key=len)
    g   = g.subgraph(gcc).copy()
    g   = nx.convert_node_labels_to_integers(g)
    print(f"  Email-Eu-Core  : {g.number_of_nodes()} nodes, "
          f"{g.number_of_edges()} edges")
    return g, "Email-Eu-Core"

def load_power_grid():
    path = "./power.gml"
    if not os.path.exists(path):
        print("  Power Grid not found.")
        print("  Download power.zip from:")
        print("  http://www-personal.umich.edu/"
              "~mejn/netdata/power.zip")
        return None, "Power Grid (Western US)"
    try:
        g = nx.read_gml(path, label=None)
        g = nx.Graph(g)
        g = nx.convert_node_labels_to_integers(g)
        gcc = max(nx.connected_components(g), key=len)
        g   = g.subgraph(gcc).copy()
        g   = nx.convert_node_labels_to_integers(g)
        print(f"  Power Grid     : {g.number_of_nodes()} nodes, "
              f"{g.number_of_edges()} edges")
        return g, "Power Grid (Western US)"
    except Exception as e:
        print(f"  Error loading power.gml: {e}")
        return None, "Power Grid (Western US)"

# ═════════════════════════════════════════════════════════════════════
# ADJACENCY MATRIX UTILITIES
# ═════════════════════════════════════════════════════════════════════

def get_adj(g, ms):
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")
    if n < ms:
        adj = sp.block_diag([adj, csr_matrix((ms-n, ms-n))])
    elif n > ms:
        adj = adj[:ms, :ms]
    return adj.tocsr()

def get_adj_T(g, ms):
    return get_adj(g, ms).T.tocsr()

def get_adj_mod(g, ms):
    n   = g.number_of_nodes()
    adj = nx.adjacency_matrix(
              g, nodelist=list(range(n))).astype("float32")
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    adj_mod = sp.diags(1.0 / deg) @ adj
    if n < ms:
        adj_mod = sp.block_diag(
                      [adj_mod, csr_matrix((ms-n, ms-n))])
    elif n > ms:
        adj_mod = adj_mod[:ms, :ms]
    return adj_mod.tocsr()

def to_padded_array(cent_dict, n_real, n_padded):
    """Always returns array of size n_padded.
    Only fills entries where node index < n_padded."""
    arr = np.zeros(n_padded, dtype=np.float32)
    for node, val in cent_dict.items():
        if 0 <= node < n_padded:   # changed n_real to n_padded
            arr[node] = float(val)
    return arr

# ═════════════════════════════════════════════════════════════════════
# EVALUATE ONE SUBGRAPH
# ═════════════════════════════════════════════════════════════════════

def evaluate_subgraph(sg, bet_model, close_model):
    sg_n = sg.number_of_nodes()

    true_bet   = nx.betweenness_centrality(
                     sg, normalized=True)
    true_close = nx.closeness_centrality(sg)

    # Pad to correct model size for each model
    true_bet_arr   = to_padded_array(
                         true_bet,   sg_n, BET_MODEL_SIZE)
    true_close_arr = to_padded_array(
                         true_close, sg_n, CLOSE_MODEL_SIZE)

    # Separate adjacency matrices for each model size
    adj_bet   = get_adj(sg,     BET_MODEL_SIZE)
    adj_T_bet = get_adj_T(sg,   BET_MODEL_SIZE)
    adj_close = get_adj(sg,     CLOSE_MODEL_SIZE)
    adj_mod   = get_adj_mod(sg, CLOSE_MODEL_SIZE)

    adj_bet_t   = sparse_mx_to_torch_sparse_tensor(
                      adj_bet,   device)
    adj_T_bet_t = sparse_mx_to_torch_sparse_tensor(
                      adj_T_bet, device)
    adj_close_t = sparse_mx_to_torch_sparse_tensor(
                      adj_close, device)
    adj_mod_t   = sparse_mx_to_torch_sparse_tensor(
                      adj_mod,   device)

    bet_kt   = np.nan
    close_kt = np.nan

    with torch.no_grad():

        # Betweenness — uses BET_MODEL_SIZE
        if bet_model is not None:
            y_bet = bet_model(adj_bet_t, adj_T_bet_t)
            if not torch.isnan(y_bet).any():
                tv_bet = torch.from_numpy(
                             true_bet_arr).float().to(device)
                kt = ranking_correlation(
                         y_bet, tv_bet,
                         sg_n, BET_MODEL_SIZE)
                if not np.isnan(kt):
                    bet_kt = kt

        # Closeness — uses CLOSE_MODEL_SIZE
        if close_model is not None:
            y_close = close_model(adj_close_t, adj_mod_t)
            if not torch.isnan(y_close).any():
                tv_close = torch.from_numpy(
                               true_close_arr).float().to(device)
                kt = ranking_correlation(
                         y_close, tv_close,
                         sg_n, CLOSE_MODEL_SIZE)
                if not np.isnan(kt):
                    close_kt = kt

    return bet_kt, close_kt

# ═════════════════════════════════════════════════════════════════════
# EVALUATE ONE GRAPH
# ═════════════════════════════════════════════════════════════════════

def evaluate_graph(g, graph_name, bet_model, close_model):
    n = g.number_of_nodes()
    print(f"\n  Graph : {graph_name}")
    print(f"  Size  : {n} nodes, {g.number_of_edges()} edges")

    # Use the larger model size as the splitting threshold
    THRESHOLD = max(BET_MODEL_SIZE, CLOSE_MODEL_SIZE)

    if n <= THRESHOLD:
        # Graph fits in model — evaluate directly
        print(f"  Evaluating directly (N <= {THRESHOLD})")
        bet_kt, close_kt = evaluate_subgraph(
                               g, bet_model, close_model)
        bet_kts   = [bet_kt]   if not np.isnan(bet_kt)   else []
        close_kts = [close_kt] if not np.isnan(close_kt) else []

    else:
        # Graph too large — split into overlapping windows
        print(f"  Graph larger than {THRESHOLD} nodes.")
        print(f"  Splitting into overlapping subgraphs...")
        nodes  = list(g.nodes())
        stride = THRESHOLD // 2
        bet_kts   = []
        close_kts = []
        n_proc    = 0

        for start in range(0, len(nodes), stride):
            subset = nodes[start:start+THRESHOLD]
            if len(subset) < 50:
                continue
            sub = g.subgraph(subset).copy()
            sub = nx.convert_node_labels_to_integers(sub)
            if not nx.is_connected(sub):
                gcc = max(nx.connected_components(sub),
                          key=len)
                sub = sub.subgraph(gcc).copy()
                sub = nx.convert_node_labels_to_integers(sub)
            if sub.number_of_nodes() < 50:
                continue

            bkt, ckt = evaluate_subgraph(
                           sub, bet_model, close_model)
            if not np.isnan(bkt):
                bet_kts.append(bkt)
            if not np.isnan(ckt):
                close_kts.append(ckt)
            n_proc += 1
            if n_proc % 10 == 0:
                print(f"  Processed {n_proc} subgraphs...")

        print(f"  Total subgraphs : {n_proc}")

    bet_mean   = np.mean(bet_kts)   if bet_kts   else np.nan
    bet_std    = np.std(bet_kts)    if bet_kts   else np.nan
    close_mean = np.mean(close_kts) if close_kts else np.nan
    close_std  = np.std(close_kts)  if close_kts else np.nan

    if not np.isnan(bet_mean):
        print(f"  Betweenness KT : "
              f"{bet_mean:.4f} +/- {bet_std:.4f}")
    else:
        print(f"  Betweenness KT : N/A")

    if not np.isnan(close_mean):
        print(f"  Closeness KT   : "
              f"{close_mean:.4f} +/- {close_std:.4f}")
    else:
        print(f"  Closeness KT   : N/A")

    return {
        "graph"         : graph_name,
        "nodes"         : n,
        "edges"         : g.number_of_edges(),
        "bet_kt_mean"   : bet_mean,
        "bet_kt_std"    : bet_std,
        "close_kt_mean" : close_mean,
        "close_kt_std"  : close_std,
    }

# ═════════════════════════════════════════════════════════════════════
# LOAD DATASETS
# ═════════════════════════════════════════════════════════════════════
print("\nLoading real-world datasets...")

datasets = []
g1, n1 = load_celegans()
if g1: datasets.append((g1, n1))

g2, n2 = load_email_eucore()
if g2: datasets.append((g2, n2))

g3, n3 = load_power_grid()
if g3: datasets.append((g3, n3))

if not datasets:
    print("\nNo datasets found.")
    print("Download the datasets and place them in "
          "your repo folder.")
    exit()

print(f"\nEvaluating on {len(datasets)} network(s)...\n")

# ═════════════════════════════════════════════════════════════════════
# EVALUATE ALL DATASETS
# ═════════════════════════════════════════════════════════════════════
results = []
for g, name in datasets:
    r = evaluate_graph(g, name, bet_model, close_model)
    results.append(r)

# ═════════════════════════════════════════════════════════════════════
# PRINT RESULTS TABLE
# ═════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  REAL-WORLD VALIDATION RESULTS")
print(f"{'='*65}")
print(f"  {'Network':<25} {'Nodes':>6} "
      f"{'Bet KT':>15} {'Close KT':>15}")
print(f"  {'-'*63}")
for r in results:
    b = (f"{r['bet_kt_mean']:.4f}+/-{r['bet_kt_std']:.4f}"
         if not np.isnan(r['bet_kt_mean']) else "N/A")
    c = (f"{r['close_kt_mean']:.4f}+/-"
         f"{r['close_kt_std']:.4f}"
         if not np.isnan(r['close_kt_mean']) else "N/A")
    print(f"  {r['graph']:<25} {r['nodes']:>6} "
          f"{b:>15} {c:>15}")

# ═════════════════════════════════════════════════════════════════════
# GENERATE FIGURE
# ═════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.dpi"  : 300,
    "font.family" : "serif",
    "font.size"   : 11,
    "axes.grid"   : True,
    "grid.alpha"  : 0.3,
})

graph_names = [r["graph"] for r in results]
bet_means   = [r["bet_kt_mean"]   for r in results]
bet_stds    = [r["bet_kt_std"]    for r in results]
close_means = [r["close_kt_mean"] for r in results]
close_stds  = [r["close_kt_std"]  for r in results]

x     = np.arange(len(graph_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

bars1 = ax.bar(
    x - width/2,
    [m if not np.isnan(m) else 0 for m in bet_means],
    width,
    yerr=[s if not np.isnan(s) else 0 for s in bet_stds],
    label="Betweenness", color="#1f77b4",
    capsize=5, edgecolor="black", linewidth=0.5
)
bars2 = ax.bar(
    x + width/2,
    [m if not np.isnan(m) else 0 for m in close_means],
    width,
    yerr=[s if not np.isnan(s) else 0 for s in close_stds],
    label="Closeness", color="#2ca02c",
    capsize=5, edgecolor="black", linewidth=0.5
)

for bar, val in zip(bars1, bet_means):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{val:.3f}", ha="center",
                va="bottom", fontsize=9)

for bar, val in zip(bars2, close_means):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{val:.3f}", ha="center",
                va="bottom", fontsize=9)

# Synthetic baseline reference lines
ax.axhline(y=0.851, color="#1f77b4", linestyle="--",
           linewidth=1.0, alpha=0.5,
           label="Synthetic baseline bet (0.851)")
ax.axhline(y=0.894, color="#2ca02c", linestyle="--",
           linewidth=1.0, alpha=0.5,
           label="Synthetic baseline close (0.894)")

ax.set_ylabel("Kendall tau rank correlation")
ax.set_title(
    "Zero-Shot GNN Performance on Real-World Networks\n"
    "(Betweenness: N=5000 model, "
    "Closeness: N=200 model, no retraining)"
)
ax.set_xticks(x)
ax.set_xticklabels(graph_names, rotation=10, ha="right")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()

fig_path = f"{OUTPUT_DIR}/fig_real_world_validation.png"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fig_path}")

# ═════════════════════════════════════════════════════════════════════
# SAVE TEXT RESULTS
# ═════════════════════════════════════════════════════════════════════
txt_path = f"{OUTPUT_DIR}/real_world_results.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    f.write("Real-World Network Validation Results\n")
    f.write("="*55 + "\n\n")
    f.write("Models:\n")
    f.write(f"  Betweenness: {BET_MODEL_FILE} "
            f"(N={BET_MODEL_SIZE})\n")
    f.write(f"  Closeness  : {CLOSE_MODEL_FILE} "
            f"(N={CLOSE_MODEL_SIZE})\n")
    f.write(f"  Mode       : Zero-shot (no retraining)\n\n")
    f.write("Synthetic baselines:\n")
    f.write("  Betweenness tau : 0.851 +/- 0.011\n")
    f.write("  Closeness tau   : 0.894 +/- 0.011\n\n")
    for r in results:
        f.write(f"{r['graph']}\n")
        f.write(f"  Nodes    : {r['nodes']}\n")
        f.write(f"  Edges    : {r['edges']}\n")
        if not np.isnan(r['bet_kt_mean']):
            f.write(f"  Bet KT   : "
                    f"{r['bet_kt_mean']:.4f} "
                    f"+/- {r['bet_kt_std']:.4f}\n")
        else:
            f.write(f"  Bet KT   : N/A\n")
        if not np.isnan(r['close_kt_mean']):
            f.write(f"  Close KT : "
                    f"{r['close_kt_mean']:.4f} "
                    f"+/- {r['close_kt_std']:.4f}\n")
        else:
            f.write(f"  Close KT : N/A\n")
        f.write("\n")

print(f"  Saved: {txt_path}\n")
print(f"{'='*65}")
print(f"  Done. Paste numbers from real_world_results.txt")
print(f"  into Chapter 4 Section on real-world validation.")
print(f"{'='*65}\n")
