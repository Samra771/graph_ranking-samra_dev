import networkx as nx
import matplotlib.pyplot as plt
import random

class Graph:

    def __init__(self, num_nodes, sparsity_index):

        self.num_nodes = num_nodes
        self.sparsity_index = sparsity_index

    def create_graph(self, graph_type):

        match graph_type:

            case 1: # Barabasi-Albert graph
                avg_rank = int((self.sparsity_index * self.num_nodes) / 2)
                graph = nx.barabasi_albert_graph(self.num_nodes, avg_rank)
                return graph

            case 2: # Erdos-Renyi graph
                graph = nx.erdos_renyi_graph(self.num_nodes, self.sparsity_index, directed=True)
                return graph

    def draw_graph(self, G, node_color='lightblue', edge_color='gray', layout='spring', node_size=300, with_labels=True):
        """
        Disegna un grafo NetworkX con opzioni di personalizzazione.

        :param G: Il grafo NetworkX da disegnare.
        :param node_color: Colore dei nodi (default: 'lightblue').
        :param edge_color: Colore degli archi (default: 'gray').
        :param layout: Layout del grafo ('spring', 'circular', 'kamada_kawai', 'random').
        :param node_size: Dimensione dei nodi.
        :param with_labels: Se True, mostra le etichette dei nodi.
        """

        # Selezione del layout
        layouts = {
            'spring': nx.spring_layout(G),  # Simulazione a molle (buono per grafi generici)
            'circular': nx.circular_layout(G),  # Disposizione circolare
            'kamada_kawai': nx.kamada_kawai_layout(G),  # Ottimizza le distanze tra i nodi
            'random': nx.random_layout(G)  # Posizioni casuali
        }

        pos = layouts.get(layout, nx.spring_layout(G))  # Default: 'spring'

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=with_labels, node_color=node_color, edge_color=edge_color, node_size=node_size)
        plt.show(block=True)

    def get_full_adjacency_matrix(self, graph):

        adj_matrix = nx.adjacency_matrix(graph)

        return adj_matrix

    def get_betweenness_centrality(self, graph):

      betweenness_centrality = nx.betweenness_centrality(graph)

      return betweenness_centrality

    def get_graph_nodes(self, graph):

        nodes = graph.nodes()

        return nodes