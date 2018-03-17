# -*- coding: utf-8 -*-

import json
import random
import sys

import click
import tqdm

from destined import evaluate_generator, lookup


@click.group()
def main():
    pass


@main.command()
@click.argument('specification-file', type=click.File('r'))
@click.argument('seed-file', type=click.File('r'))
@click.argument('output-file', type=click.File('w'))
def evaluate(specification_file, seed_file, output_file):

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

    # Generating function for input seed values.
    def seeds():
        while True:
            line = seed_file.readline()
            if not line:
                break
            try:
                seed = int(line)
            except ValueError:
                seed = hash(line)
            yield seed

    # Mapper.
    delayed = map(sample, seeds())

    # Run the thing.
    for entry in tqdm.tqdm(delayed):
        json.dump(entry, output_file)
        output_file.write('\n')

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
