# -*- coding: utf-8 -*-

__author__ = 'Simon Bowly'
__email__ = 'simon.bowly@gmail.com'
__version__ = '0.1.0'

import functools

from .distribution import evaluate_distribution
from .generators import lookup

evaluate_generator = functools.partial(
    evaluate_distribution, function_lookup=lookup)
