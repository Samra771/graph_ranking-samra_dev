# GNN Centrality Approximation

**Graph Neural Networks for Efficient Node Centrality Approximation in Complex Networks**

Implementation of Graph Neural Networks for fast approximation of betweenness
and closeness centrality in complex networks, developed as part of the research
paper:

> Samra Sana and Giorgio Mantica,
> "Graph Neural Networks for Efficient Node Centrality Approximation
> in Complex Networks,"
> Università degli Studi dell'Insubria, 2026.

This implementation is adapted from the original GNN_Ranking repository by
Maurya et al. (2021): https://github.com/sunilkmaurya/GNN_Ranking

---

## Overview

Computing betweenness or closeness centrality exactly requires O(N³) operations,
making it infeasible for large networks. This repository trains Graph Neural
Networks to approximate both centrality measures by learning from examples,
reducing inference at test time to a single forward pass through a fixed-depth
message-passing architecture.

The betweenness GNN uses a dual-pathway architecture:
- Pathway 1 processes the adjacency matrix A (outgoing geodesic flow)
- Pathway 2 processes the transpose A^T (incoming geodesic flow)
- Scores from both pathways are multiplied elementwise to produce the final ranking

The closeness GNN uses a single pathway with degree-normalised adjacency
preprocessing and 7 message-passing layers, trained on undirected graphs
to avoid degenerate zero-centrality values.

---

## Differences From Original Maurya et al. Repository

| Aspect | Original (Maurya et al.) | This Implementation |
|---|---|---|
| Network size | 5,000–10,000 nodes | 200 nodes |
| Training graphs | 40 | 1,600 (betweenness), 4,000 (closeness) |
| Test graphs | 10 | 400 (betweenness), 1,000 (closeness) |
| Learning rate | 0.0005 | 0.0001 (betweenness), 0.0005 (closeness) |
| Epochs | 15 | 100 (betweenness), 50 (closeness) |
| Dropout | 0.6 | 0.2 |
| Loss function | Pairwise ranking loss | Pairwise ranking loss (unchanged) |
| Graph types | SF, ER, GRP | ER, BA, GRP (mixed training supported) |
| Closeness model | Included | Refactored with degree-normalised adjacency |
| Logging | None | TensorBoard |
| Model saving | Not implemented | Saved as .pth with timestamp |
| Experiment scripts | Single training script | Full experiment pipeline (see below) |

---

## Requirements

Python 3.10 or higher is required (the code uses the `match` statement
introduced in Python 3.10).

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages:
- torch >= 2.0
- networkx >= 3.0
- scipy >= 1.10
- numpy >= 1.24
- matplotlib >= 3.7
- tensorboard >= 2.12

---

---

## Reproducing Paper Results

All experiments in the paper can be reproduced by running the scripts below
in order. Each script saves figures to a dedicated output folder and prints
a results summary to the terminal.

### Step 1 — Train the betweenness model

```bash
python run_experiment.py
```

Settings: 2,000 ER graphs, N = 200 nodes, p = 0.15, 100 epochs, lr = 1e-4.
Outputs saved to `./paper_figures/`. Runtime: ~30–60 min on CPU.

### Step 2 — Train the closeness model

```bash
python run_experiment_closeness.py
```

Settings: 5,000 undirected ER graphs, N = 200 nodes, p = 0.15, 50 epochs, lr = 5e-4.
Outputs saved to `./paper_figures_closeness/`. Runtime: ~60–90 min on CPU.

### Step 3 — Baseline comparison

```bash
python experiment1_baselines.py
```

Compares GNN against random ranking, degree centrality, and Brandes approximation
(k = 10, 20, 50) on 200 test graphs. Requires trained models from Steps 1 and 2.
Outputs saved to `./paper_figures_baselines/`.

### Step 4 — Generalization across graph types

```bash
python experiment2_generalization.py
```

Tests ER-trained models on Barabási-Albert and Gaussian Random Partition graphs.
Trains a mixed model on all three graph types and evaluates cross-topology
performance. Outputs saved to `./paper_figures_generalization/`.

### Step 5 — Scalability and timing

```bash
python experiment3_scalability_fixed.py
```

Measures GNN inference time vs exact NetworkX computation across
N in {50, 100, 200, 500, 1000}. Outputs saved to `./paper_figures_scalability/`.

### Step 6 — Degree centrality comparison on structured graphs

```bash
python experiment4_degree_vs_gnn.py
```

Measures degree centrality performance on BA and GRP graphs alongside GNN,
demonstrating where the GNN advantage is clearest. Outputs saved to
`./paper_figures_degree_comparison/`.

---

## Key Results

| Experiment | Result |
|---|---|
| Betweenness (ER test) | Kendall τ = 0.855 ± 0.008 |
| Closeness (ER test) | Kendall τ = 0.840 ± 0.011 |
| Betweenness Mixed → BA | Kendall τ = 0.933 ± 0.005 |
| Betweenness Mixed → GRP | Kendall τ = 0.879 ± 0.010 |
| Speedup at N = 200 | 14× faster than NetworkX |
| Speedup at N = 1000 | 152× faster than NetworkX |

---

## Interactive Training with TensorBoard

For interactive training with real-time loss monitoring:

```bash
python graph_ranking_train_main.py
```

In a separate terminal, launch TensorBoard:

```bash
tensorboard --logdir="./logs" --load_fast=false
```

Open http://localhost:6006 in your browser to view live training curves
including training loss, test loss, relative loss difference, and Kendall τ.

---

## Prediction on a New Graph

To load a trained model and predict centrality on a new graph:

```bash
python graph_ranking_predict_main.py
```

You will be prompted to select the graph type, graph size, and saved model file.

---

## Loss Function

Both models are trained using a **pairwise ranking loss** (MarginRankingLoss,
margin = 1.0). For each graph, N × 20 node pairs are sampled and the loss
penalises violations of the correct pairwise ranking order. This directly
optimises node ranking rather than absolute centrality values, which is the
correct objective for centrality approximation.

## Evaluation Metric

Model quality is assessed using **Kendall's τ rank correlation coefficient**,
which measures how well the GNN preserves the relative ordering of nodes by
centrality. This is the appropriate metric for centrality approximation:
correctly identifying the most influential nodes matters more than predicting
exact numerical values.

---

## Graph Types

| Code | Type | Notes |
|---|---|---|
| 1 | Barabási-Albert | Scale-free; used in generalization experiments |
| 2 | Erdős-Rényi | Random; primary training graph type |
| 9 | Random (mixed) | Randomly selects type 1 or 2 per graph |

GRP (Gaussian Random Partition) graphs are generated directly via NetworkX
in the experiment scripts and are not available through the interactive CLI.

---

## Citation

If you use this code in your research, please cite both the original
architecture paper and this implementation:

**Original architecture:**
S. K. Maurya, X. Liu, and T. Murata,
"Graph neural networks for fast node ranking approximation,"
ACM Transactions on Knowledge Discovery from Data,
vol. 15, no. 5, pp. 1–32, 2021.

**This implementation:**
S. Sana and G. Mantica,
"Graph Neural Networks for Efficient Node Centrality Approximation
in Complex Networks,"
Università degli Studi dell'Insubria, 2026.
https://github.com/Samra771/graph_ranking-samra_dev