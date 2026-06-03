# GNN Centrality Approximation

**Graph Neural Networks for Efficient Node Centrality
Approximation in Complex Networks**

> Samra Sana, Giorgio Mantica, and Saul Imbrici 
> Università degli Studi dell'Insubria, 2026.

Adapted from Maurya et al. (2021):
https://github.com/sunilkmaurya/GNN_Ranking

---

## What This Repository Does

Computing betweenness and closeness centrality exactly
requires O(N³) operations, prohibitively slow for large
networks. This repository trains Graph Neural Networks to
approximate both measures from graph structure alone,
reducing inference to a single forward pass regardless of
network size.

Two GNN models are implemented. The betweenness model uses
a dual-pathway architecture that processes the adjacency
matrix and its transpose in parallel, capturing both outgoing
and incoming geodesic flow. The closeness model uses a single
pathway with degree-normalised adjacency preprocessing.
Both are trained using pairwise ranking loss, which directly
optimises node ordering rather than absolute centrality values.

Models trained on synthetic Erdős-Rényi graphs generalize
to Barabási-Albert scale-free networks and community-structured
graphs without retraining. A dedicated model trained at
N = 5,000 nodes achieves Kendall τ = 0.938 with exact labels,
and GNN inference is 91× faster than exact computation at
N = 1,000 nodes.

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch 2.0+, NetworkX 3.0+

---

## Reproducing Results

Run scripts in this order:

```bash
# Train primary models
python run_experiment.py                  # betweenness (N=200)
python run_experiment_closeness.py        # closeness (N=200)

# Hyperparameter selection
python hyperparameter_search.py           # betweenness grid search
python hyperparameter_search_closeness.py # closeness grid search

# Experiments
python experiment1_baselines.py           # vs random, degree, Brandes
python experiment2_generalization.py      # ER → BA, GRP, mixed training
python experiment3_scalability_fixed.py   # timing across N=50 to 1000
python experiment4_degree_vs_gnn.py       # degree centrality comparison

# Large-scale training (GPU required)
python train_N5000_fixed.py               # betweenness at N=5,000
```

Each script saves figures and a numerical results summary to
its own output folder. Full instructions and expected outputs
are documented in the paper.

---

## Key Results

| Experiment | Kendall tau |
|---|---|
| Betweenness (N=200, ER test) | 0.851 ± 0.011 |
| Closeness (N=200, ER test) | 0.894 ± 0.011 |
| Betweenness (N=5,000, exact labels) | 0.938 ± 0.001 |
| Mixed-trained → BA (scale-free) | 0.920 ± 0.006 |
| Mixed-trained → GRP (community) | 0.861 ± 0.012 |
| Inference speedup at N=1,000 | 91× vs NetworkX exact |

---
