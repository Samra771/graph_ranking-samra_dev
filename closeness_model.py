"""
closeness_model.py
==================
GNN model for closeness centrality approximation.

Architecture differences from betweenness model:
- Single pathway (processes A only, not A and A^T)
- 7 GNN layers instead of 5
- Output is direct sum of scores from all 7 layers (no elementwise product)
- Scores are summed, not multiplied across pathways

This reflects the mathematical structure of closeness centrality:
closeness measures how quickly a node reaches others (receptivity),
which depends on outgoing connections only — hence single pathway.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GNN_Layer, GNN_Layer_Init, MLP
import time


class GNN_Close(nn.Module):

    def __init__(self, ninput, nhid, dropout, learning_rate, weight_decay):
        super(GNN_Close, self).__init__()

        # 7 GNN layers — two more than betweenness
        # Closeness depends on longer-range path structure
        self.gc1 = GNN_Layer_Init(ninput, nhid)   # Initial embedding layer
        self.gc2 = GNN_Layer(nhid, nhid)
        self.gc3 = GNN_Layer(nhid, nhid)
        self.gc4 = GNN_Layer(nhid, nhid)
        self.gc5 = GNN_Layer(nhid, nhid)
        self.gc6 = GNN_Layer(nhid, nhid)
        self.gc7 = GNN_Layer(nhid, nhid)

        self.dropout    = dropout
        self.score_layer = MLP(nhid, self.dropout)

        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay

    def forward(self, adj, adj_mod):
        """
        Forward pass — single pathway for closeness.

        adj     : standard adjacency matrix A
        adj_mod : column-masked adjacency matrix (removes nodes not on any geodesic)

        Layer structure:
          gc1 uses adj      (initial embedding from raw adjacency)
          gc2..gc7 use adj_mod  (message passing through masked adjacency)

        L2 normalization is applied at every layer except the last.
        Scores from all 7 layers are summed to produce the final ranking.
        """

        # ── Message passing ───────────────────────────────────────────────────
        x_1 = F.normalize(F.relu(self.gc1(adj)),            p=2, dim=1)
        x_2 = F.normalize(F.relu(self.gc2(x_1, adj_mod)),   p=2, dim=1)
        x_3 = F.normalize(F.relu(self.gc3(x_2, adj_mod)),   p=2, dim=1)
        x_4 = F.normalize(F.relu(self.gc4(x_3, adj_mod)),   p=2, dim=1)
        x_5 = F.normalize(F.relu(self.gc5(x_4, adj_mod)),   p=2, dim=1)
        x_6 = F.normalize(F.relu(self.gc6(x_5, adj_mod)),   p=2, dim=1)
        x_7 = F.relu(self.gc7(x_6, adj_mod))   # No normalization on last layer

        # ── Score computation — one score per layer ───────────────────────────
        score_1 = self.score_layer(x_1, self.dropout)
        score_2 = self.score_layer(x_2, self.dropout)
        score_3 = self.score_layer(x_3, self.dropout)
        score_4 = self.score_layer(x_4, self.dropout)
        score_5 = self.score_layer(x_5, self.dropout)
        score_6 = self.score_layer(x_6, self.dropout)
        score_7 = self.score_layer(x_7, self.dropout)

        # ── Final output — direct sum (no cross-pathway product) ──────────────
        output = (score_1 + score_2 + score_3 +
                  score_4 + score_5 + score_6 + score_7)

        return output

    def model_to_device(self, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device), device

    def get_optimizer(self, model):
        return torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def save_model(self, model):
        model_name = f"closeness_{time.strftime('%Y%m%d-%H%M%S')}.pth"
        torch.save(model.state_dict(), model_name)
        print(f"  Model saved as {model_name}")
        return model_name
