# Learning Transferable Centrality Representations with Graph Neural Networks

**Samra Sana, Giorgio Mantica, and Saul Imbrici**
Università degli Studi dell'Insubria, 2026

Implementation accompanying the paper:

> *Learning Transferable Centrality Representations with Graph Neural Networks: Generalization Across Network Topologies*

---

## Overview

This repository implements Graph Neural Networks (GNNs) for approximating **betweenness** and **closeness centrality** directly from graph structure.

The models are trained on synthetic networks using pairwise ranking loss and evaluated on:

* Erdős–Rényi (ER) graphs
* Barabási–Albert (BA) scale-free graphs
* Gaussian Random Partition (GRP) community graphs
* Real-world networks

The study focuses on **generalization across graph topologies**, demonstrating that training on structurally diverse graph families improves transferability and robustness.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements:

* Python 3.10+
* PyTorch 2.0+
* NetworkX 3.0+

## Reproducing Experiments

```bash
# Train models
python run_experiment.py
python run_experiment_closeness.py

# Hyperparameter search
python hyperparameter_search.py
python hyperparameter_search_closeness.py

# Evaluation
python experiment1_baselines.py
python experiment2_generalization.py
python experiment3_scalability_fixed.py
python experiment4_degree_vs_gnn.py

# Large-scale experiment (N = 5000)
python train_N5000_fixed.py
```

---

## Main Results

| Experiment                   | Kendall τ     |
| ---------------------------- | ------------- |
| Betweenness (ER)             | 0.861 ± 0.011 |
| Closeness (ER)               | 0.894 ± 0.011 |
| Mixed-trained → BA           | 0.920 ± 0.006 |
| Mixed-trained → GRP          | 0.861 ± 0.012 |
| Betweenness (N = 5000)       | 0.938 ± 0.001 |
| Inference speedup (N = 1000) | 91×           |

---

## Citation

If you use this code, please cite the associated paper.

```text
Sana, S., Mantica, G., and Imbrici, S.
Learning Transferable Centrality Representations with Graph Neural Networks:
Generalization Across Network Topologies, 2026.
```
