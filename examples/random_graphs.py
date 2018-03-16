
import functools
import itertools
import multiprocessing
import random

import networkx as nx
import pandas as pd
from tqdm import tqdm

from destined import evaluate_distribution


def undirected_noloop_erdos_renyi_np(randgen, n, p):
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(
        edge
        for edge in itertools.combinations(range(n), 2)
        if randgen.uniform(0, 1) < p)
    return g


lookup = {
    'graphs.undirected.erdos_renyi_np': undirected_noloop_erdos_renyi_np,
    'choice': random.Random.choice,
    'uniform': random.Random.uniform,
}


evaluate = functools.partial(
    evaluate_distribution, function_lookup=lookup.__getitem__)


generate = evaluate({
    "generator": "graphs.undirected.erdos_renyi_np",
    "parameters": {
        "n": {
            "generator": "choice",
            "parameters": {
                "seq": {"value": [10, 20, 50, 100]}
            },
        },
        "p": {
            "generator": "uniform",
            "parameters": {
                "a": {"value": 0},
                "b": {"value": 0.6}
            },
        }
    },
})


def features(g):
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    return {
        'nodes': nodes,
        'edges': edges,
        'density': 2 * edges / (nodes * (nodes - 1)),
        'connected': nx.is_connected(g)
    }


def sample(seed):
    return features(generate(random.Random(seed)))


if __name__ == '__main__':

    pool = multiprocessing.Pool()
    map = pool.imap_unordered

    sysrandom = random.SystemRandom()
    seeds = [
        sysrandom.getrandbits(32)
        for _ in range(100000)]

    tasks = map(sample, seeds)
    results = pd.DataFrame(list(tqdm(tasks, total=len(seeds))))
    results.to_feather('random-graphs.feather')
