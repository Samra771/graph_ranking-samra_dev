# graph_ranking
# GNN Centrality Approximation

Implementation of Graph Neural Networks for fast approximation of betweenness centrality in complex networks.

This code is adapted from the original GNN_Ranking repository by Maurya et al. (2021):
https://github.com/sunilkmaurya/GNN_Ranking

**Paper:** "Graph Neural Networks for Efficient Node Centrality Approximation in Complex Networks"
samra sana, Università degli Studi dell'Insubria, 2026

---

## What This Code Does

Given a graph, computing betweenness centrality exactly requires O(N³) operations — too slow for large networks. This code trains a Graph Neural Network to approximate betweenness centrality by learning from examples, reducing inference to a single forward pass through 5 message-passing layers.

The GNN uses a dual-pathway architecture:
- One pathway processes the adjacency matrix A (capturing outgoing geodesic flow)
- One pathway processes the transpose A^T (capturing incoming geodesic flow)
- Scores from both pathways are multiplied elementwise to produce the final centrality ranking

---

## Differences From Original Maurya et al. Repository

| Aspect | Original | This Implementation |
|--------|----------|---------------------|
| Network size | 5,000–10,000 nodes | 200 nodes |
| Training graphs | 40 | 1,600 |
| Test graphs | 10 | 400 |
| Learning rate | 0.0005 | 0.0001 |
| Epochs | 15 | 100 |
| Dropout | 0.6 | 0.2 |
| Interface | argparse CLI | Interactive + script |
| Logging | None | TensorBoard |
| Graph types | SF, ER, GRP | BA, ER |
| Model saving | Not implemented | Saved as .pth |

---

## Requirements

Python 3.10+ is required (uses `match` statement).

Install all dependencies:
```bash
pip install -r requirements.txt
```

Main packages needed:
- torch >= 2.0
- networkx >= 3.0
- scipy >= 1.10
- numpy >= 1.24
- matplotlib >= 3.7
- tensorboard >= 2.12

---

## File Structure

```
graph_ranking-samra_dev/
│
├── run_experiment.py              ← START HERE: trains model + generates paper figures
│
├── graph_ranking_train_main.py    ← Interactive training with TensorBoard logging
├── graph_ranking_predict_main.py  ← Load saved model and predict on new graph
│
├── betweennes_model.py            ← GNN architecture (5-layer dual-pathway)
├── betweennes_training_library.py ← Training loop, loss function, evaluation
├── graph_mgmt_library.py          ← Graph creation and betweenness computation
├── graph_utils.py                 ← Utilities: shuffling, splitting, file I/O
├── layer.py                       ← GNN layer definitions (GNN_Layer, MLP)
│
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## Quick Start — Generate Paper Figures

To train the model with the settings used in the paper (200 nodes, 2000 graphs, 100 epochs) and generate all figures automatically:

```bash
python run_experiment.py
```

This will:
1. Generate 2,000 Erdős-Rényi graphs with 200 nodes and edge probability p = 0.15
2. Split into 1,600 training graphs and 400 test graphs
3. Train the GNN for 100 epochs
4. Save the trained model as `betweenness_model.pth`
5. Save all paper figures to `./paper_figures/`:
   - `fig_training_loss.png` — training loss per epoch
   - `fig_test_loss.png` — test loss per epoch
   - `fig_relative_loss_diff.png` — relative loss difference per epoch
   - `fig_kendall_tau.png` — Kendall tau rank correlation per epoch
   - `fig_training_grid.png` — combined 2x2 figure (for paper Figure 4.1)
   - `fig_betweenness_comparison.png` — GNN vs NetworkX comparison (for paper Figure 4.2)
   - `results_summary.txt` — numerical results for all epochs

Expected runtime: approximately 30–60 minutes on CPU, 5–10 minutes on GPU.

---

## Interactive Training (with TensorBoard)

For interactive training with real-time loss monitoring:

```bash
python graph_ranking_train_main.py
```

Then in a separate terminal, start TensorBoard to visualise training:

```bash
tensorboard --logdir="./logs" --load_fast=false
```

Open your browser at http://localhost:6006 to see live training curves.

---

## Prediction on a New Graph

After training, to run the model on a new graph:

```bash
python graph_ranking_predict_main.py
```

You will be prompted to choose the graph type, size, and which saved model to load.

---

## Loss Function

The model is trained using a **pairwise ranking loss** (MarginRankingLoss with margin=1.0), not MSE. For each graph, N×20 node pairs are sampled and the loss penalises violations of the correct pairwise ranking order. This directly optimises node ranking rather than absolute centrality values.

---

## Evaluation Metric

Model quality is measured using **Kendall's τ rank correlation coefficient**, which measures how well the GNN preserves the relative ordering of nodes by centrality. This is the correct metric for centrality approximation since identifying the most influential nodes correctly matters more than predicting exact numerical values.

---

## Graph Types Supported

| Code | Type | Notes |
|------|------|-------|
| 1 | Barabási-Albert | Scale-free, directed |
| 2 | Erdős-Rényi | Random, directed — used in paper |
| 9 | Random (mixed) | Randomly selects type 1 or 2 |

---

## Citation

If you use this code, please cite both the original paper and this implementation:

**Original architecture:**
S. K. Maurya, X. Liu, and T. Murata, "Graph neural networks for fast node ranking approximation," ACM TKDD, vol. 15, no. 5, pp. 1–32, 2021.

**This implementation:**
M. A. Rigamonti, "GNN Centrality Approximation — Adapted Implementation," GitHub, 2026.
https://github.com/Samra771/graph_ranking-samra_dev
