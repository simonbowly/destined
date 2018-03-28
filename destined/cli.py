# -*- coding: utf-8 -*-

import contextlib
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
    # TODO add --parallel option here run the zmq setup.


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
@click.option('--port', type=int, default=5560)
def request_worker(specification_file, port):
    ''' Worker process which loads a specification at runtime, then requests
    work until told to stop. '''
    sample_function = get_sample_function(json.load(specification_file))
    context = zmq.Context()
    tasker = context.socket(zmq.REQ)
    tasker.connect("tcp://localhost:{:d}".format(port))
    tasker.send(b'ready')
    while True:
        message = tasker.recv()
        if message == b'stop':
            break
        elif message == b'hold':
            time.sleep(0.5)
            tasker.send(b'ready')
        else:
            response = worker_response(sample_function, message)
            tasker.send(response)
    tasker.close()
    context.destroy()
    return 0


@contextlib.contextmanager
def launch_request_workers(specification, nworkers, port):
    # Launch worker processes and write specification to their stdin.
    workers = []
    for i in range(nworkers):
        worker = subprocess.Popen(
            [
                'destined', 'request_worker', '-',
                '--port={:d}'.format(port)],
            encoding='utf-8', start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,)
        worker.stdin.write(specification)
        worker.stdin.close()
        workers.append(worker)
        click.echo(f'Worker {i+1} PID {worker.pid}', err=True)
    try:
        # Provide the worker process list in the context.
        yield workers
    finally:
        # Wait for the workers to exit (user should have signalled 'done').
        for worker in workers:
            try:
                worker.wait(timeout=1)
            except subprocess.TimeoutExpired:
                worker.send_signal(signal.SIGINT)
            worker.wait()
            click.echo(f'Worker on {worker.pid} exited with return code {worker.returncode}.', err=True)
            # TODO print stderr if unexpected return code.


# CLICK DECORATOR ON A GENERATOR SHOULD PROBABLY THROW AN ERROR!!!


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('samples', type=int)
@click.argument('output-file', type=click.File('w'))
@click.option('--chunk', type=int, default=10)
@click.option('--nworkers', type=int, default=4)
@click.option('--port', type=int, default=5560)
@click.option('--progress/--no-progress', default=True, help='Show progress bar')
def parallel_router(specification_file, samples, output_file, chunk, nworkers, port, progress):
    specification = specification_file.read()
    results = gen_parallel_router(specification, samples, chunk, nworkers, port)
    if progress:
        results = tqdm.tqdm(results, total=samples, smoothing=0)
    for result in results:
        json.dump(result, output_file)
        output_file.write('\n')
    return 0


def gen_parallel_router(specification, samples, chunk, nworkers, port):

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind("tcp://*:{:d}".format(port))

    with launch_request_workers(specification, nworkers, port):

        iter_seeds = system_random_seeds(samples)
        issued_tasks = 0
        received_results = 0

        while issued_tasks == 0 or received_results < issued_tasks:

            # Pull a message off the queue.
            identity, mid, message = socket.recv_multipart()
            assert mid == b''

            # Send new work back to this worker, or shut it down.
            seeds = list(itertools.islice(iter_seeds, chunk))
            if len(seeds) > 0:
                socket.send_multipart([identity, b'', packb(seeds)])
                issued_tasks += 1
            else:
                click.echo(f'Sent stop signal to {identity}', err=True)
                socket.send_multipart([identity, b'', b'stop'])

            # Worker is either signalling ready or returning a result.
            if message == b'ready':
                click.echo(f'Received ready signal from {identity}', err=True)
            else:
                # TODO assert we have issued a task to this identity before.
                results = unpackb(message)
                for result in results:
                    yield result
                received_results += 1

    socket.close()
    context.destroy()


@main.command()
@click.option('--port', type=int, default=5560)
def router(port):
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind("tcp://*:{:d}".format(port))

    ntasks = 10
    issued_tasks = 0
    received_results = 0
    workers = set()

    while received_results < ntasks:
        # Collect a message.
        click.echo(f'Issued: {issued_tasks}   Received: {received_results}')
        print(workers)
        # Make this recv() a poll with timeout. If timeout, check state of
        # the workers (use context block again). Could reissue a lost task
        # (unless all workers died).
        identity, mid, message = socket.recv_multipart()
        workers.add(identity)
        assert mid == b''
        # Process message.
        if message == b'ready':
            click.echo('Received ready signal')
        else:
            # Check if worker reports an error (should be part of worker_reponse)
            # in which case we should kill off the parallel evaluation.
            click.echo(f'Received {len(unpackb(message))} results.')
            received_results += 1
        # Send new work back to this worker, or shut it down.
        if issued_tasks < ntasks:
            socket.send_multipart([identity, b'', packb(list(range(100)))])
            issued_tasks += 1
        else:
            # socket.send_multipart([identity, b'', b'stop'])
            socket.send_multipart([identity, b'', b'hold'])
            # YES WE CAN LOSE WORKERS HERE!!!
            # MUST ENSURE QUEUE IS EMPTY

        # Catch interrupt signals, change mode to just issue 'stop' commands.
        # If we get a socket timeout in this mode, just exit. Context block
        # should kill off the workers. Their sockets won't be in a bad state
        # (REQ failed to issue) because the bound socket is still open.

    # introduces the race condition
    time.sleep(0.49)

    # clear the queue of any waiting messages, reissuing hold
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    while True:
        socks = dict(poller.poll(timeout=0.01))
        if socket in socks:
            ident, _, msg = socket.recv_multipart()
            assert msg == b'ready'
            socket.send_multipart([ident, b'', b'hold'])
        else:
            # Poller timeout = queue is temporarily empty.
            # Close before anyone else can register.
            socket.close()
            break

    # Having to rely on this (and not entirely solving the race condition,
    # because this code is quite literally racing and relying on timings)
    # makes this seem like a bad pattern for loosely bound systems. PUB-SUB
    # would be much better "here is some work I want done, whoever is out
    # there, send me whatever results you like". i.e. publish all tasks, workers
    # can choose a random one, continue until every task is executed at lease
    # once/

    context.destroy()
    return 0

'''
Advantages/disadvantages.

ventilate -> pull -> push -> sink
Good because the worker can't end up stuck. It's waiting state is passively
listening for new tasks, so if the sink drops messages we just lose results,
not workers. The sink can read and ignore results without replying, workers
can stay up and will continue on their merry way.
Bad because of the slow joiner issue.
This is probably good for use cases where the user is actively involved in
starting workers - we just keep ventilating tasks until the required number
of results is achieved. We are assuming there that seeds were dropped due
to random worker death ... if it turns out to be due to slow response then
we have a sample bias issue (in the random sampling use case).

req -> router
Good because it's load balanced. Workers actively request new tasks.
Bad because if the router drops messages we 'forget' that workers are sitting
there in a 'ready' state. We have to remember to send 'stop' signals or the
REQ sockets can end up locked.
This is good for self contained use cases, but may be bad for keeping workers up.

Alternative - rathen than sending b'stop', send b'hold'. This causes the worker
to wait a bit, then reissue its ready request. This holds workers open when there
is no work and others are still finishing up. Then at the end, you want something
like this:

# All work is now done, every worker should have been issued a
# hold signal.
while True:
    socks = dict(poller.poll(timeout=0.1))
    if socket in socks:
        ident, _, msg = socket.recv_multipart()
        assert msg == b'Ready'
        socket.send_multipart([ident, b'', b'hold'])
    else:
        # Poller timeout = queue is temporarily empty.
        socket.close()

Now that the socket is closed, workers send(b'Ready') will block
until a new ROUTER socket issues 'bind' and gets their messages. That
router will immediately know about all workers UNLESS there is a race
condition where the socket picks up a message in between poll() and close().
That message is then lost because no new 'hold' was issued, and since the
worker cannot repeat its b'ready' message without first receiving a reply,
it can never be detected by a new router.

Fuzzing works to show the race condition here, so presumably it is always
possible unless it could be somehow an atomic zmq operation.
But the approach of reissuing the 'hold' signal should do the trick since
those messages go out very quickly compared to the delay worker-side.

NB context.destroy() (and normal process exit) becomes blocked when a message
was sent but not delivered. So a REQ socket can exit safely if it has sent
a request and not received a reply, but it cannot exit safely if no socket
was bound on the port to allow the message to be delivered. Presumably this
does not apply to the bind side?

'''


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--sub-port', type=int, default=5559)
@click.option('--host', type=str, default='localhost')
def source_sink_worker(specification_file, source_port, sink_port, sub_port, host):
    ''' Worker process which loads a generator specification at runtime, then
    pulls seed values from a source and pushes results to a sink. '''
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.connect(f"tcp://{host}:{source_port:d}")
    sender = context.socket(zmq.PUSH)
    sender.connect(f"tcp://{host}:{sink_port:d}")
    subscriber = context.socket(zmq.SUB)
    # Localhost controls shutdown
    subscriber.connect(f"tcp://localhost:{sub_port:d}")
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
    click.echo("Press Enter when workers are ready...")
    _ = input()
    # Send the number of tasks distributed to the sink as start signal.
    sink.send(packb(samples))
    # Send random seeds to workers in chunks.
    iter_seeds = system_random_seeds(samples)
    while True:
        seeds = list(itertools.islice(iter_seeds, chunk))
        if len(seeds) == 0:
            break
        sender.send(packb(seeds))
    sink.close()
    # Stay running so others don't bind this port.
    try:
        while True:
            signal.pause()
    finally:
        sender.close()
        context.destroy()


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
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--progress/--no-progress', default=True, help='Show progress bar')
def collect(samples, output_file, source_port, sink_port, progress):
    ''' Launch a seed ventilator and result collector. This is a blind process
    which does not consider the specification, just sends seeds to workers and
    collects whatever they return. '''
    # Run the sink internally for result collection.
    context = zmq.Context()
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(sink_port))
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
    return 0


@contextlib.contextmanager
def launch_workers(context, nworkers, specification, source_port, sink_port, sub_port, host):
    ''' Launches worker processes. Uses pub-sub channel to signal worker
    shutdown. All communication on the publisher socket only occurs within
    this context, so publisher can be closed. '''
    # Start a publisher socket to signal workers to stop gracefully.
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://*:{:d}".format(sub_port))
    # Launch worker processes and write specification to their stdin.
    workers = []
    for i in range(nworkers):
        worker = subprocess.Popen(
            [
                '/home/simon/.virtualenvs/py36/bin/destined',
                'source_sink_worker', '-',
                f'--source-port={source_port:d}',
                f'--sink-port={sink_port:d}',
                f'--sub-port={sub_port:d}',
                f'--host={host}'],
            encoding='utf-8', start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,)
        worker.stdin.write(specification)
        worker.stdin.close()
        workers.append(worker)
        click.echo(f'Worker {i+1} PID {worker.pid}', err=True)
    try:
        # Provide the worker process list in the context.
        yield workers
    finally:
        # Publish shutdown signal to workers and wait for them to exit.
        publisher.send(b'')
        for worker in workers:
            worker.wait()
            click.echo(f'Worker on {worker.pid} exited with return code {worker.returncode}.')
            # TODO print stderr if unexpected return code.
        publisher.close()


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('nworkers', type=int)
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--sub-port', type=int, default=5559)
@click.option('--host', type=str, default='localhost')
def run_workers(specification_file, nworkers, source_port, sink_port, sub_port, host):
    ''' Launch and manage multiple workers with a given specification. '''
    context = zmq.Context()
    with launch_workers(
            context, nworkers, specification_file.read(),
            source_port, sink_port, sub_port, host) as workers:
        click.echo("Workers up. Ctrl+C to shut down.")
        try:
            signal.pause()
            #while True:
            #    time.sleep(1)
            #    click.echo(str({
            #        worker.pid: worker.poll()
            #        for worker in workers}))
        except KeyboardInterrupt:
            click.echo("Shutting down workers.")
    context.destroy()
    click.echo("Done.")
    return 0


@contextlib.contextmanager
def launch_ventilator(samples, chunk, source_port, sink_port):
    ventilator = subprocess.Popen(
        [
            'destined', 'ventilator',
            str(samples), str(chunk),
            f'--source-port={source_port:d}',
            f'--sink-port={sink_port:d}'],
        encoding='utf-8',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    # click.echo(f'Ventilator PID {ventilator.pid}', err=True)
    try:
        yield ventilator
    finally:
        ventilator.send_signal(signal.SIGINT)
        ventilator.wait()
        # click.echo(f'Ventilator on {ventilator.pid} exited with return code {ventilator.returncode}.')


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('samples', type=int)
@click.argument('output-file', type=click.File('w'))
@click.option('--chunk', type=int, default=10)
@click.option('--nworkers', type=int, default=4)
@click.option('--source-port', type=int, default=5557)
@click.option('--sink-port', type=int, default=5558)
@click.option('--sub-port', type=int, default=5559)
@click.option('--progress/--no-progress', default=True, help='Show progress bar')
def parallel(specification_file, samples, output_file, chunk, nworkers,
             source_port, sink_port, sub_port, progress):
    ''' Launch a seed ventilator and result collector. This is a blind process
    which does not consider the specification, just sends seeds to workers and
    collects whatever they return. '''
    results = parallel_generator(
        specification_file.read(), samples, chunk, nworkers,
        source_port, sink_port, sub_port)
    if progress:
        results = tqdm.tqdm(results, smoothing=0, total=samples)
    try:
        for result in results:
            json.dump(result, output_file)
            output_file.write('\n')
    except ValueError as e:
        click.echo(str(e), err=True)


def parallel_generator(specification, samples, chunk, nworkers,
                       source_port, sink_port, sub_port):
    context = zmq.Context()
    # Sink runs in this process to collect results, ventilator and workers
    # are launched in separate processes.
    receiver = context.socket(zmq.PULL)
    receiver.bind("tcp://*:{}".format(sink_port))
    poller = zmq.Poller()
    poller.register(receiver, zmq.POLLIN)
    with launch_workers(
            context, nworkers, specification,
            source_port, sink_port, sub_port) as workers:
        with launch_ventilator(samples, chunk, source_port, sink_port):
            # Wait for synchronisation from the ventilator process.
            message = receiver.recv()
            assert unpackb(message) == samples
            # Collect results from sink port until total samples reached.
            collected = 0
            while collected < samples:
                socks = dict(poller.poll(timeout=100))
                if receiver in socks:
                    # Got a new message, expand it out to single results.
                    message = receiver.recv()
                    decoded = unpackb(message)
                    for result in decoded:
                        yield result
                        collected += 1
                else:
                    # Workers have been quiet a while, check they are ok.
                    dead_workers = [
                        worker.pid
                        for worker in workers
                        if worker.poll() is not None]
                    if dead_workers:
                        raise ValueError(f"Workers on PIDs {dead_workers} died!")
    # Nesting of contexts means that the program is dismantled in the order
    # ventilator -> workers -> collector, and nothing can be left in these
    # communication pipelines. PUB-SUB pattern doesn't leave remnants (?) so
    # this is not so much of an issue (handled in the worker context anyway).
    receiver.close()
    context.destroy()


# TODO handling dead workers. The problem here is that ventilated seeds are queued
# on the worker, so a dead worker means we lose a portion of the tasks. Because the
# tasks are random, there is no real need to establish which tasks were dropped, so
# we could just restart the ventilator. Any excess seeds will be discarded when the
# collector reaches the required number of samples and kills them usng the pub-sub
# channel. This is probably a rare enough case that we don't need to bother? If
# workers die due to an error then we probably don't need to keep throwing tasks
# at the others.

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


if __name__ == '__main__':
    main()

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
