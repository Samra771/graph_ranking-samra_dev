"""
graph_mgmt_library.py  (updated — adds closeness centrality support)
=====================
Adds two new methods to the Graph class:
  get_closeness_centrality(graph)      — computes exact closeness centrality
  get_column_masked_adjacency(graph)   — builds the adj_mod matrix for GNN_Close
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
import random


class Graph:

    def __init__(self, num_nodes, sparsity_index):
        self.num_nodes     = num_nodes
        self.sparsity_index = sparsity_index

    # ── Graph creation ────────────────────────────────────────────────────────

    def create_graph(self, graph_type):
        match graph_type:
            case 1:  # Barabasi-Albert
                avg_rank = max(1, int((self.sparsity_index * self.num_nodes) / 2))
                return nx.barabasi_albert_graph(self.num_nodes, avg_rank)
            case 2:  # Erdos-Renyi (directed)
                return nx.erdos_renyi_graph(
                    self.num_nodes, self.sparsity_index, directed=True
                )

    # ── Adjacency matrices ────────────────────────────────────────────────────

    def get_full_adjacency_matrix(self, graph):
        """Standard adjacency matrix — used by both models."""
        return nx.adjacency_matrix(graph)

    def get_column_masked_adjacency(self, graph):
        """
        Builds adj_mod for the closeness GNN.

        In the betweenness model we remove ROWS (nodes not on any outgoing geodesic).
        In the closeness model we remove COLUMNS (nodes not on any incoming geodesic).

        This reflects the fact that closeness centrality measures how quickly
        a node can REACH others — so we care about which nodes are reachable
        (columns), not which nodes are starting points (rows).

        How it works:
          1. Get the standard adjacency matrix A
          2. Compute in-degree and out-degree for each node
          3. Nodes with zero in-degree OR zero out-degree are not on any geodesic
          4. Zero out the columns corresponding to those nodes in A
          5. Return the column-masked matrix as adj_mod
        """
        adj = nx.adjacency_matrix(graph).astype(np.float32)
        adj_dense = np.array(adj.todense())

        # Compute out-degree (row sums) and in-degree (column sums)
        out_degree = adj_dense.sum(axis=1)   # shape (N,)
        in_degree  = adj_dense.sum(axis=0)   # shape (N,)

        # Nodes that participate in geodesics must have both in and out connections
        participates = ((out_degree > 0) & (in_degree > 0)).astype(np.float32)

        # Apply column mask — zero columns of non-participating nodes
        adj_mod = adj_dense * participates[np.newaxis, :]

        return csr_matrix(adj_mod)

    # ── Centrality measures ───────────────────────────────────────────────────

    def get_betweenness_centrality(self, graph):
        """
        Exact betweenness centrality via NetworkX.
        Returns dict: {node_id: value}
        Normalized=True means values are in [0, 1].
        """
        return nx.betweenness_centrality(graph, normalized=True)

    def get_closeness_centrality(self, graph):
        """
        Exact closeness centrality via NetworkX.
        Returns dict: {node_id: value}

        For directed graphs, NetworkX computes closeness as:
          c(u) = (n-1) / sum of shortest path lengths from u to all reachable nodes
        Nodes that cannot reach any other node get centrality 0.

        This is the u_closeness_centrality (outgoing closeness).
        """
        return nx.closeness_centrality(graph)

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_graph_nodes(self, graph):
        return graph.nodes()

    def draw_graph(self, G, node_color='lightblue', edge_color='gray',
                   layout='spring', node_size=300, with_labels=True):
        layouts = {
            'spring'      : nx.spring_layout(G),
            'circular'    : nx.circular_layout(G),
            'kamada_kawai': nx.kamada_kawai_layout(G),
            'random'      : nx.random_layout(G)
        }
        pos = layouts.get(layout, nx.spring_layout(G))
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=with_labels,
                node_color=node_color, edge_color=edge_color,
                node_size=node_size)
        plt.show(block=True)
