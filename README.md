Wosis - a python package developed to support analysis of Web of Science data from querying to visualization.

[![DOI](https://zenodo.org/badge/155658135.svg)](https://zenodo.org/badge/latestdoi/155658135)

The scope of this package is quite narrow and is currently intended for limited use.

Examples of its use can be found by looking at the notebooks within the
[sd-prac-bibanalysis](https://github.com/ConnectedSystems/sd-prac-bibanalysis) repository.

Key dependencies include:

* WOS Client, a SOAP-based client for Web of Science, developed by E. Bacis [@enricobacis](https://github.com/enricobacis)
* wos_parser, a parser for Web of Science XML data, developed by T. Achakulvisut [@titipata](https://github.com/titipata)
* Metaknowledge, a Python library for bibliometric research, developed at [Networks Lab](https://github.com/networks-lab/metaknowledge)

For the moment it is probably best to install by:

```bash
$ git clone https://github.com/ConnectedSystems/wosis.git
$ cd wosis
$ python setup.py develop
```

Alternatively, via `pip`

```bash
$ pip install git+https://github.com/ConnectedSystems/wosis.git@master
```
