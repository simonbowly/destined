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

from destined import evaluate_generator, lookup


packb = functools.partial(msgpack.packb, use_bin_type=True)
unpackb = functools.partial(msgpack.unpackb, raw=False)


def get_sample_func(specification_file):

    # Load specification from inputs.
    specification = json.load(specification_file)

    # Generating function taking a randgen object and returning an instance.
    generate = evaluate_generator(specification['instances'])

    # Function to evaluate features of the generated instance.
    attributes = lookup(specification['attributes'])

    # Sample function returns a data point given a seed value.
    def sample(seed):
        randgen = random.Random(seed)
        instance = generate(randgen)
        return dict(attributes(instance), seed=seed)

    return sample


@click.group()
def main():
    pass


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('output-file', type=click.File('w'))
@click.option('--samples', type=int, default=1000)
@click.option('--progress/--no-progress', default=True)
def evaluate(specification_file, output_file, samples, progress):
    ''' Serial processor. '''

    # Sampling function built from specification.
    sample = get_sample_func(specification_file)

    # System random seeds for the given count.
    sysrandom = random.SystemRandom()
    system_seeds = [sysrandom.getrandbits(64) for _ in range(samples)]

    # Mapper.
    delayed = map(sample, system_seeds)
    if progress:
        delayed = tqdm.tqdm(delayed, smoothing=0)

    # Run the thing.
    for entry in delayed:
        json.dump(entry, output_file)
        output_file.write('\n')

    return 0


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.option('--port', type=int, default=5555)
def reply_worker(specification_file, port):
    ''' Worker using a reply pattern. '''

    # Sampling function built from specification.
    sample = get_sample_func(specification_file)

    # Bind the target socket and start the loop.
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:{}".format(port))

    while True:

        # Receive a task.
        message = socket.recv()
        decoded = unpackb(message)
        seeds = decoded   # input error handling here
        # click.echo(f"Processing {len(seeds)} seeds.")

        # Sample for each seed.
        results = [sample(seed) for seed in seeds]

        # Pack and send.
        encoded = packb(results)
        socket.send(encoded)


@main.command()
@click.argument('output-file', type=click.File('w'))
@click.option('--samples', type=int, default=1000)
@click.option('--chunk', type=int, default=100)
@click.option('--port', type=int, default=5555)
def request_client(output_file, samples, chunk, port):
    ''' Client using a request pattern. '''

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:{}".format(port))

    sysrandom = random.SystemRandom()
    system_seeds = [sysrandom.getrandbits(64) for _ in range(samples)]
    iter_seeds = iter(system_seeds)

    def wrapper():

        while True:

            # Get a chunk of seed data.
            seeds = list(itertools.islice(iter_seeds, chunk))
            if len(seeds) == 0:
                break

            # Send seed values as jobs.
            encoded = packb(seeds)
            socket.send(encoded)

            # Get the reply.
            message = socket.recv()
            decoded = unpackb(message)

            # Pass results back to the consumer.
            for result in decoded:
                yield result

    delayed = wrapper()

    # Run the thing.
    for entry in tqdm.tqdm(delayed, smoothing=0):
        json.dump(entry, output_file)
        output_file.write('\n')


@main.command()
@click.argument('output-file', type=click.File('w'))
@click.option('--port', type=int, default=5558)
@click.option('--continuous/--no-continuous', default=False)
@click.option('--progress/--no-progress', default=True)
def sink(output_file, port, continuous, progress):

    context = zmq.Context()

    # Socket to receive messages on.
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(port))

    def wrapper():
        while True:
            message = receiver.recv()
            decoded = unpackb(message)
            for result in decoded:
                yield result

    while True:
        # Wait for start of batch signal from ventilator.
        # This signal tells the sink how many results to expect (but
        # nothing about how they are batched).
        message = receiver.recv()
        count = unpackb(message)

        # Receive a set number of results.
        delayed = itertools.islice(wrapper(), count)
        if progress:
            delayed = tqdm.tqdm(delayed, smoothing=0, total=count)
        for entry in delayed:
            json.dump(entry, output_file)
            output_file.write('\n')

        if not continuous:
            break


@main.command()
@click.argument('port', type=int)
@click.argument('output_file', type=click.File('w'))
def clean_sink(port, output_file):
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(port))
    while True:
        message = receiver.recv()
        sys.stdout.write('.')
        sys.stdout.flush()


@main.command()
@click.argument('samples', type=int)
@click.argument('chunk', type=int)
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
def ventilator(samples, chunk, source_port, sink_port):

    context = zmq.Context()

    # Socket to send messages on.
    sender = context.socket(zmq.PUSH)
    sender.bind("tcp://*:{}".format(source_port))

    # Socket with direct access to the sink: used to syncronize start of batch.
    sink = context.socket(zmq.PUSH)
    sink.connect("tcp://localhost:{}".format(sink_port))

    time.sleep(1)
    # click.echo("Press Enter when the workers are ready: ")
    # _ = input()
    # click.echo("Sending tasks to workers")

    # The first message is "0" and signals start of batch
    encoded = packb(samples)
    sink.send(encoded)

    # Create the seed data.
    sysrandom = random.SystemRandom()
    system_seeds = [sysrandom.getrandbits(64) for _ in range(samples)]
    iter_seeds = iter(system_seeds)

    # Send in chunks.
    while True:
        seeds = list(itertools.islice(iter_seeds, chunk))
        if len(seeds) == 0:
            break
        encoded = packb(seeds)
        sender.send(encoded)

    # Give 0MQ time to deliver
    time.sleep(1)


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
def push_pull_worker(specification_file, source_port, sink_port):

    sample = get_sample_func(specification_file)

    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:{}".format(source_port))

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:{}".format(sink_port))

    # Process tasks forever
    while True:

        # Receive a task.
        message = receiver.recv()
        decoded = unpackb(message)
        seeds = decoded   # input error handling here
        # click.echo(f"Processing {len(seeds)} seeds.")

        # Sample for each seed.
        results = [sample(seed) for seed in seeds]

        # Pack and send.
        encoded = packb(results)
        sender.send(encoded)


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('output-file', type=click.File('w'))
@click.option('--samples', type=int, prompt=True)
@click.option('--chunk', type=int, prompt=True)
@click.option('--workers', type=int, prompt=True)
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--progress/--no-progress', default=True)
def parallel(specification_file, output_file,
             samples, chunk, workers,
             source_port, sink_port,
             progress):

    # Start workers with the specification.
    str_spec = specification_file.read()
    specification = json.loads(str_spec)
    worker_processes = []
    for i in range(workers):
        worker = subprocess.Popen(
            ['destined', 'push_pull_worker', '-'],
            encoding='utf-8',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE)
        worker.stdin.write(str_spec)
        worker.stdin.close()
        click.echo(f'Worker {i+1} PID {worker.pid}', err=True)
        worker_processes.append(worker)

    # Run the sink in this process.
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(sink_port))

    def wrapper():
        while True:
            message = receiver.recv()
            decoded = unpackb(message)
            for result in decoded:
                yield result

    # Start the ventilator externally.
    # This could be internal, i.e.
    # 1. start workers
    # 2. connect sink
    # 3. ventilate
    # 4. empty sink
    # Only issue is if the ventilator blocks for some
    # reason?
    ventilator = subprocess.Popen(
        ['destined', 'ventilator', str(samples), str(chunk)],
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    click.echo(f'Ventilator PID {ventilator.pid}', err=True)

    # Collect in sink.
    message = receiver.recv()
    count = unpackb(message)
    delayed = itertools.islice(wrapper(), count)
    if progress:
        delayed = tqdm.tqdm(delayed, smoothing=0, total=count)
    for entry in delayed:
        json.dump(entry, output_file)
        output_file.write('\n')

    # Once done the workers can be killed off.
    for worker in worker_processes:
        worker.send_signal(signal.SIGINT)
        worker.wait()

    # This should already be in an exited state, but
    # make sure it completes.
    ventilator.wait()



# make this a common cli decorator
# @click.argument('specification-file', type=click.File('r'))
# sample = get_sample_func(specification_file)

# Run sink and ventilator as async tasks of the same process.
# That way the ventilator can dispatch based on what the sink
# has received, monitor progress together, etc, without extra
# processes?
# This could also verify that all dispatched tasks were recieved,
# and monitor any missed seeds.
# But this feels more like a router approach anyway...

# How to check if a port is 'clean' so that there are no workers
# with a different specification loaded which could leak messages.

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
