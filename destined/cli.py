# -*- coding: utf-8 -*-

import sys
import click


@click.command()
def main(args=None):
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
