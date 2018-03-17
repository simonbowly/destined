
import itertools
import random

import networkx as nx


def undirected_noloop_erdos_renyi_np(randgen, nodes, prob):
    g = nx.Graph()
    g.add_nodes_from(range(nodes))
    g.add_edges_from(
        edge
        for edge in itertools.combinations(range(nodes), 2)
        if randgen.uniform(0, 1) < prob)
    return g


def uniform(randgen, low, high):
    return randgen.uniform(low, high)


def graph_features(g):
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    return {
        'nodes': nodes,
        'edges': edges,
        'density': 2 * edges / (nodes * (nodes - 1)),
        'connected': nx.is_connected(g)
    }


directory = {
    'graphs.undirected_noloop_erdos_renyi_np': undirected_noloop_erdos_renyi_np,
    'graphs.features': graph_features,
    'choice': random.Random.choice,
    'uniform': uniform,
}


lookup = directory.__getitem__
