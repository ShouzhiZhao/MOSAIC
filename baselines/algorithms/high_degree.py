def high_degree(G, k):
    return [n for n, d in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:k]]
