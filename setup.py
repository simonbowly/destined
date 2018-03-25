#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'click>=6.0',
    'networkx>=2.1',
    'tqdm>=4.19.7',
    'msgpack>=0.5.6',
    'pyzmq>=17.0'
    ]

setup_requirements = [
    ]

test_requirements = [
    'pytest',
    'mock',
    ]

setup(
    author="Simon Bowly",
    author_email='simon.bowly@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Declarative specification for test instance distributions.",
    entry_points={
        'console_scripts': [
            'destined=destined.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    keywords='destined',
    name='destined',
    packages=find_packages(include=['destined']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/simonbowly/destined',
    version='0.1.0',
    zip_safe=False,
)
