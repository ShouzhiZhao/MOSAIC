def degree_discount(G, k, p=0.1):
    S = []
    d = dict(G.degree())
    dd = d.copy()
    t = dict()

    for u in G.nodes():
        t[u] = 0

    for _ in range(k):
        if not dd:
            break
        u = max(dd, key=dd.get)
        S.append(u)
        dd.pop(u)

        neighbors = list(G.successors(u))

        for v in neighbors:
            if v in dd:
                t[v] += 1
                weight = G[u][v].get("weight", p)
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * weight

    return S
