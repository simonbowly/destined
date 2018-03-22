
Destined
========

DEclarative Specification for Test INstancE Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Declare a distribution of instances by specifying function parameters.

.. code:: json

    {
        "generator": "graphs.erdos_renyi",
        "parameters":
        {
            "nodes": {"value": 100},
            "edges": {"value": 2500}
        }
    }

Nest distribution specifications to vary parameters.

.. code:: json

    {
        "generator": "graphs.erdos_renyi",
        "parameters":
        {
            "nodes": {"value": 100},
            "edges":
            {
                "generator": "randint",
                "parameters":
                {
                    "low": {"value": 100},
                    "high": {"value": 4000}
                }
            }
        }
    }

Sample from the distribution using system seeds.

.. code:: sh

    destined evaluate examples/random-3sat.json 1000 -


* Free software: MIT license
