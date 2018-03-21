
import io
import itertools
import random
import re
import subprocess

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


def randint(randgen, low, high):
    return randgen.randint(low, high)


def graph_features(g):
    nodes = g.number_of_nodes()
    edges = g.number_of_edges()
    return {
        'nodes': nodes,
        'edges': edges,
        'density': 2 * edges / (nodes * (nodes - 1)),
        'connected': nx.is_connected(g)
    }


def random_k_sat(randgen, nvariables, nclauses, clause_length):
    var_choice = list(range(1, nvariables + 1))
    neg_choice = [1, -1]
    clauses = [
        [
            variable * randgen.choice(neg_choice)
            for variable in randgen.sample(var_choice, clause_length)
        ]
        for _ in range(nclauses)
    ]
    return dict(nvariables=nvariables, clauses=clauses)


def write_dimacs(instance, outfile):
    ''' Write DIMACS CNF to the open file/buffer. '''
    outfile.write('p cnf {} {}\n'.format(
        instance['nvariables'], len(instance['clauses'])))
    outfile.writelines(
        ' '.join(str(v) for v in variables) + ' 0\n'
        for variables in instance['clauses'])


def run_clasp(instance, timeout=None):
    ''' Pipe the DIMACS CNF representation of the SAT instance to
    the clasp solver. Return stdout and stderr streams. The optional
    timeout argument (in seconds) is passed to clasp with the
    --time-limit=<n> option.

    Usage:
        returncode, stdout, stderr = run_clasp(sat_instance)
        returncode, stdout, stderr = run_clasp(sat_instance, timeout=30)
    '''
    command = ['clasp']
    if timeout is not None:
        command.append('--time-limit={:d}'.format(timeout))
    process = subprocess.Popen(
        command, encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE)
    write_dimacs(instance, process.stdin)
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def parse_clasp_stdout(output):
    result = output.split('\n')
    result = [line for line in result if line.startswith('s')]
    assert len(result) == 1
    flag, sat = result[0].split()
    sat = sat.lower()
    assert flag == 's'
    info = {
        'clasp_time': float(re.search(
            'c +Time +: +([0-9\.]+)s',
            output).group(1)),
        'clasp_cpu_time': float(re.search(
            'c +CPU Time +: +([0-9\.]+)s',
            output).group(1))}
    if sat == 'satisfiable':
        return dict(satisfiable=True, **info)
    elif sat == 'unsatisfiable':
        return dict(satisfiable=False, **info)
    else:
        return dict(satisfiable=None, **info)


def sat_features(instance):
    _, stdout, _ = run_clasp(instance)
    features = parse_clasp_stdout(stdout)
    features.update(dict(
        nvariables=instance['nvariables'],
        nclauses=len(instance['clauses'])))
    return features


directory = {
    'graphs.undirected_noloop_erdos_renyi_np': undirected_noloop_erdos_renyi_np,
    'graphs.features': graph_features,
    'sat.uniform_k_sat': random_k_sat,
    'sat.features': sat_features,
    'choice': random.Random.choice,
    'randint': randint,
    'uniform': uniform,
}


lookup = directory.__getitem__


if __name__ == '__main__':

    import sys

    instance = random_k_sat(random.Random(), 800, 10000, 3)
    returncode, stdout, stderr = run_clasp(instance, timeout=30)

    print('*** INSTANCE ***')
    with open('instance.cnf', 'w') as outfile:
        write_dimacs(instance, outfile)
    write_dimacs(instance, sys.stdout)

    print('\n*** CLASP ***')
    print(f'Return code: {returncode}\n')
    print(stdout)
    print(stderr)

    print('*** RESULT ***')
    print(parse_clasp_stdout(stdout))
