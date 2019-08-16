import hashlib

from .convert import rc_to_df
import wos_parser
import json

from os.path import join as pj
from datetime import datetime

import pickle

__all__ = ['create_query_hash', 'export_ris_file', 'store_query_hash', 
           'export_representative_file', 'combine_manually_sorted',
           'write_cache', 'read_cache']

def create_query_hash(query_str, time_span=None):
    hash_object = hashlib.md5(query_str.encode())
    md5_hash = hash_object.hexdigest()

    if time_span:
        md5_hash = "{}_{}-{}".format(md5_hash, time_span['begin'], time_span['end'])

    return md5_hash
# End create_query_hash()


def write_cache(recs, cache_file):
    with open(cache_file, 'wb') as outfile:
        pickle.dump(recs, outfile, pickle.HIGHEST_PROTOCOL)
    # End with
# End write_cache()


def read_cache(cache_file):
    with open(cache_file, 'rb') as infile:
        recs = pickle.load(infile)
    # End with

    return recs
# End read_cache()


def export_ris_file(records, filename):
    """Write out records to RIS formatted file.

    Parameters
    ==========
    * records : Metaknowledge RecordCollection
    * filename : str, path and filename.
    """
    records.writeFile(filename)
# End export_ris_file()


def store_query_hash(hash_to_query, fn='hash_to_query.txt'):
    # use `json.loads` to do the reverse
    with open(fn, 'w') as file:
         file.write(json.dumps(hash_to_query, indent=2))
    # End with
# End store_query_hash()


def export_representative_file(records, retrieval_date, data_fn='../data/repset.csv'):
    """Write out representative dataset with proprietary WoS data removed.

    Parameters
    ==========
    * records : Metaknowledge RecordCollection
    * retrieval_date : str, Date of retrieval
    * data_fn : str, folder and filename to export data to (folder must exist first!)
    """
    repset_df = rc_to_df(records)

    assert len(repset_df.id.unique()) == len(repset_df.id), "Duplicate records found!"

    # Removing proprietary data
    hide_columns = ['abstract', 'keywords', 'id', 'kws', 'CR']
    hide_columns = repset_df.columns.intersection(hide_columns)
    repset_df = repset_df.drop(hide_columns, axis=1)

    gen_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(data_fn, 'w') as fn:
        fn.write("# Data from Clarivate Analytics' Web of Science, retrieved {}\n".format(retrieval_date))
        fn.write("# This file generated on {}\n".format(gen_date))
        repset_df.index.name = "item"
        repset_df.to_csv(fn)
    # End with
# End export_representative_file()


def combine_manually_sorted(target, other):
    """Combine two DataFrames representing manually sorted records.

    Parameters
    ==========
    * target : DataFrame, data will be merged into this DataFrame
    * other : DataFrame, data will be copied from this DataFrame

    Returns
    ==========
    * New Merged DataFrame
    """
    assert hasattr(target, 'DOI'), "Both DataFrames have to include DOIs"
    assert hasattr(other, 'DOI'), "Both DataFrames have to include DOIs"
    assert hasattr(other, 'relevant'), "The DataFrame with data to be copied across has to have a 'relevant' column"

    result = target.copy()
    result.update(other)

    return result
# End combine_manually_sorted()
