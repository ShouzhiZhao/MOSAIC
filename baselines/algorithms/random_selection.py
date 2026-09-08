import random


def random_selection(G, k):
    return random.sample(list(G.nodes()), k)
