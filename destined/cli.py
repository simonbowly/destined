# -*- coding: utf-8 -*-

import functools
import itertools
import json
import random
import signal
import subprocess
import sys
import time

import click
import msgpack
import tqdm
import zmq

from .distribution import evaluate_distribution
from .generators import lookup


packb = functools.partial(msgpack.packb, use_bin_type=True)
unpackb = functools.partial(msgpack.unpackb, raw=False)


def get_sample_function(specification):
    ''' Still working on this spec... read a random generator specification
    (see example files) including evaluation/attribute commands. Returns a
    function which, given a random seed, generates an instance and returns
    its attributes. '''

    # Generating function taking a randgen object and returning an instance.
    generate = evaluate_distribution(specification['instances'], lookup)

    # Function to evaluate features of the generated instance.
    attributes = lookup(specification['attributes'])

    # Sample function returns a data point given a seed value.
    def sample(seed):
        randgen = random.Random(seed)
        instance = generate(randgen)
        return dict(attributes(instance), seed=seed)

    return sample


def system_random_seeds(samples):
    sysrandom = random.SystemRandom()
    for _ in range(samples):
        yield sysrandom.getrandbits(64)


def generate_with_system_seeds(specification, samples):
    ''' Generator function yielding :samples outputs of the specification
    using system random seeds. '''
    sysrandom = random.SystemRandom()
    sample_function = get_sample_function(specification)
    for _ in range(samples):
        seed = sysrandom.getrandbits(64)
        yield sample_function(seed)


@click.group()
def main():
    pass


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('samples', type=int)
@click.argument('output-file', type=click.File('w'))
@click.option('--progress/--no-progress', default=True, help='Show progress bar')
def evaluate(specification_file, samples, output_file, progress):
    ''' Generate the given number of samples for the specification using
    system random seeds. '''
    specification = json.load(specification_file)
    results = generate_with_system_seeds(specification, samples)
    if progress:
        results = tqdm.tqdm(results, total=samples)
    for result in results:
        json.dump(result, output_file)
        output_file.write('\n')
    return 0


def worker_response(sample_function, message):
    ''' Protocol for workers. Expects a msgpack bytes encoded list of integer
    seed values. Evaluates the given sample function for these seeds, returns
    msgpack bytes encoded list of attribute dicts.
    Tests - this should raise an error if either the input is unexpected or
    the returned data from sample_function is unexpected. '''
    seeds = unpackb(message)
    assert type(seeds) is list
    assert all(type(seed) is int for seed in seeds)
    results = [sample_function(seed) for seed in seeds]
    assert all(type(result) is dict for result in results)
    return packb(results)


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--sub-port', type=int, default=5559)
def source_sink_worker(specification_file, source_port, sink_port, sub_port):
    ''' Worker process which loads a generator specification at runtime, then
    pulls seed values from a source and pushes results to a sink. '''
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:{:d}".format(source_port))
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:{:d}".format(sink_port))
    subscriber = context.socket(zmq.SUB)
    subscriber.connect("tcp://localhost:{:d}".format(sub_port))
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    # Poller to enable listening on receiver and subscriber.
    poller = zmq.Poller()
    poller.register(receiver, zmq.POLLIN)
    poller.register(subscriber, zmq.POLLIN)
    # Load the specification and listen for tasks forever.
    specification = json.load(specification_file)
    sample_function = get_sample_function(specification)
    while True:
        try:
            socks = dict(poller.poll())
        except KeyboardInterrupt:
            break
        if receiver in socks:
            message = receiver.recv()
            response = worker_response(sample_function, message)
            sender.send(response)
        if subscriber in socks:
            message = subscriber.recv()
            assert message == b''
            break
    return 0


@main.command()
@click.argument('samples', type=int)
@click.argument('chunk', type=int)
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
def ventilator(samples, chunk, source_port, sink_port):
    ''' Fire off a set number of tasks in chunks of the given size. '''
    context = zmq.Context()
    sender = context.socket(zmq.PUSH)
    sender.bind("tcp://*:{}".format(source_port))
    # Connect to the sink to synchronise batch start.
    sink = context.socket(zmq.PUSH)
    sink.connect("tcp://localhost:{}".format(sink_port))
    # Wait for everyone to connect...
    time.sleep(1)
    # Send the number of tasks distributed to the sink as start signal.
    sink.send(packb(samples))
    # Send random seeds to workers in chunks.
    iter_seeds = system_random_seeds(samples)
    while True:
        seeds = list(itertools.islice(iter_seeds, chunk))
        if len(seeds) == 0:
            break
        sender.send(packb(seeds))
    # Give 0MQ time to deliver
    time.sleep(1)
    return 0


def sink_wrapper(receiver):
    ''' Wraps the sink socket, which receives results in chunks, to produce
    a generator of single results. '''
    while True:
        message = receiver.recv()
        decoded = unpackb(message)
        for result in decoded:
            yield result


@main.command()
@click.argument('samples', type=int)
@click.argument('output-file', type=click.File('w'))
@click.option('--chunk', type=int, default=10)
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--progress/--no-progress', default=True, help='Show progress bar')
def ventilate_collect(samples, output_file, chunk,
                      source_port, sink_port, progress):
    ''' Launch a seed ventilator and result collector. This is a blind process
    which does not consider the specification, just sends seeds to workers and
    collects whatever they return. '''
    # Run the sink internally for result collection.
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(sink_port))
    # Launch the ventilator externally.
    ventilator = subprocess.Popen(
        [
            'destined', 'ventilator',
            str(samples), str(chunk),
            f'--source-port={source_port:d}',
            f'--sink-port={sink_port:d}'],
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    click.echo(f'Ventilator PID {ventilator.pid}', err=True)
    # Wait for synchronisation from the ventilator.
    message = receiver.recv()
    assert unpackb(message) == samples
    # Collect results on sink port until total is reached.
    results = itertools.islice(sink_wrapper(receiver), samples)
    if progress:
        results = tqdm.tqdm(results, smoothing=0, total=samples)
    for result in results:
        json.dump(result, output_file)
        output_file.write('\n')
    # Wait for the ventilator to terminate (should be complete well before the
    # workers finished.
    ventilator.wait()
    return 0


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('nworkers', type=int)
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--sub-port', type=int, default=5559)
def run_workers(specification_file, nworkers, source_port, sink_port, sub_port):
    ''' Launch and manage multiple workers with a given specification. '''
    # Start a publisher socket to signal workers to stop gracefully.
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:{:d}".format(sub_port))
    # Worker processes are launched by writing the specification to
    # their stdin.
    specification = specification_file.read()
    worker_processes = []
    for i in range(nworkers):
        worker = subprocess.Popen(
            [
                'destined', 'source_sink_worker', '-',
                f'--source-port={source_port:d}',
                f'--sink-port={sink_port:d}',
                f'--sub-port={sub_port:d}'],
            encoding='utf-8', start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,)
        worker.stdin.write(specification)
        worker.stdin.close()
        worker_processes.append(worker)
        click.echo(f'Worker {i+1} PID {worker.pid}', err=True)
    # Signal the user and wait for interrupt signal to shut down workers.
    click.echo("Workers up. Ctrl+C to shut down.")
    try:
        while True:
            time.sleep(1)
            # Monitor for any workers which have exited.
            for worker in worker_processes:
                returncode = worker.poll()
                if returncode is not None:
                    click.echo(f'Worker on {worker.pid} exited with return code {returncode}.')
                    worker.wait()
            # Trim any dead workers for the next pass.
            worker_processes = [worker for worker in worker_processes if worker.poll() is None]
            if len(worker_processes) == 0:
                click.echo("All workers have died.")
                break
    except KeyboardInterrupt:
        click.echo("Shutting down workers.")
    # Publish shutdown signal to workers and wait for them to exit.
    publisher.send(b'')
    for worker in worker_processes:
        worker.wait()
        click.echo(f'Worker on {worker.pid} exited with return code {worker.returncode}.')
    click.echo("Done.")
    return 0


@main.command()
@click.option('--port', type=int, default=5558)
def clean_sink(port):
    ''' Utility function to clean a pull socket if it becomes corrupted. '''
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(port))
    while True:
        message = receiver.recv()
        sys.stdout.write('.')
        sys.stdout.flush()


# TODO
#
# Combined parallel evaluation. Launch workers with start_new_session,
# launch ventilator externally, collect in main process.
# Shutdown process:
#   1. Shutdown ventilator (avoids leaving seeds in worker source pipe).
#      signal.SIGINT & wait for exit is fine here.
#   2. Publish kill signal to workers, wait for them to exit (avoids leaving
#      results in sink pipe).
#   3. Exit gracefully.
#
# socket.close() and context.destroy() with sensible finally handling.
#
# Verify workers return the right spec?
#
# Make a --parallel/--workers option for zmq parallel evaluation. Move the
# more complex cli elements to a separate script destined-parallel
#
