"""
VoteRank — Zhang et al., Sci. Rep. 6, 27823 (2016).
Iterative voting on the graph; for DiGraph, behavior matches NetworkX:
nodes vote for in-neighbors; elected node and its out-neighbors are damped.
"""

import networkx as nx


def voterank(G, k):
    """Return top-k seeds in VoteRank order (first elected = strongest)."""
    if k <= 0:
        return []
    ranked = nx.voterank(G, number_of_nodes=min(k, G.number_of_nodes()))
    return ranked[:k]
