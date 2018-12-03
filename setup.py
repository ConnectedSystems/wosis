#! /usr/bin/env python
from setuptools import setup

if __name__ == "__main__":
    setup(
        name='Wosis',
        version='0.1.1',
        description='Python package that acts as an bridging interface between WoS Client, wos_parser, and Metaknowledge',
        long_description=open('README.md').read(),
        url='',
        author='Takuya Iwanaga',
        author_email='iwanaga.takuya@anu.edu.au',
        license='(c) 2018 Takuya Iwanaga',
        packages=['wosis'],
        install_requires=[
            'lxml',
            'pyyaml',
            'wos',
            'matplotlib',
            'seaborn',
            'pandas',
            'networkx',
            'python-louvain',
            'nltk',
            'fuzzywuzzy',
            'scikit-learn',
            'tqdm'
        ],
        dependency_links=[
            'pip install git+https://github.com/titipata/wos_parser.git',
            'pip install git+https://github.com/ConnectedSystems/metaknowledge.git@add-collections'],
    )
