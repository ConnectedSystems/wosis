#! /usr/bin/env python
from setuptools import setup

if __name__ == "__main__":
    setup(
        name='Wosis',
        version='0.1.dev0',
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
            'metaknowledge',
            'matplotlib',
            'seaborn',
            'pandas',
            'networkx',
            'python-louvain',
            'nltk'
        ],
        dependency_links=['https://github.com/titipata/wos_parser/tarball/master#egg=wos_parser-0'],
    )
