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

# Quick Start

You will need access to the Premium API for Clarivate's Web of Science. This is given as a username and password.
Create a `.yml` file with these details in the following format:

```yaml
wos:
  user: username
  password: password
```

The configuration file can then be loaded like so:

```python
import wosis

path_to_your_config_file = "config.yml"
wos_config = wosis.load_config(path_to_your_config_file)
```

Then build a query using a list of desired and undesired terms and the subject areas to search. The format follows the standard given by Web of Science, as seen [here](https://images.webofknowledge.com/images/help/WOS/hs_wildcards.html).

```python
list_of_terms_to_be_included = ["some", "keywords", "of", "interest"]
excluded_terms = ["I", "do not", "want", "these", "keywords"]
subject_areas = ["ENVIRONMENTAL SCIENCES"]

# Build a list of queries to send
topics = [wosis.build_query(list_of_terms_to_be_included, excluded_terms, subject_areas), ]
```

The queries can then be sent to the Web of Science servers. The results will be dumped to a text file labelled with a `query_id` inside a temporary directory (`tmp`). Be warned that this temporary data store is up to you to manage. Please take care to remove the data once your analysis is complete.

```python
overwrite = False  # do not overwrite existing data store if it exists
id_to_query = wosis.query(topics, overwrite, wos_config)
```

`id_to_query` will be a Python dictionary which maps a generated query id to the query that was sent. Because we sent a single query in this example, we are interested in the first `query_id`.

```python
import pandas as pd
import metaknowledge as mk

query_id = list(id_to_query.keys())[0]  # Get the first query_id
RC = mk.RecordCollection(f"tmp/{query_id}.txt")  # Load the results
```

Wosis provides convenient plotting methods.

```python
import wosis.analysis.plotting as wos_plot

wos_plot.plot_kw_trend(RC, title='Plot of the number of keywords over time', save_plot_fn='figs/num_kw_per_pub.png')
```
