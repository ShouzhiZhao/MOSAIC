"""
ClusterRank-style local heuristic (weighted directed variant).
Builds on the idea of weighting neighbors by degree and local clustering;
weighted formulation aligns with Li et al., Physica A 519, 208–218 (2019).
"""

import networkx as nx


def clusterrank(G, k):
    """
    One-shot score per node: sum over outgoing edges (i->j) of
    w_ij * (out_degree(j) + 1) * (1 - CC_U(j)), where CC_U is the
    clustering coefficient on the undirected skeleton of G.
    """
    if k <= 0:
        return []
    U = G.to_undirected()
    cc = nx.clustering(U)
    out_deg = dict(G.out_degree())

    score = {}
    for i in G.nodes():
        s = 0.0
        for j in G.successors(i):
            w = float(G[i][j].get("weight", 1.0))
            cj = cc.get(j, 0.0)
            s += w * (out_deg[j] + 1.0) * (1.0 - cj)
        score[i] = s

    ordered = sorted(score.keys(), key=lambda x: score[x], reverse=True)
    return ordered[:k]
