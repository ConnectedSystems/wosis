import metaknowledge as mk
import pandas as pd


def extract_recs(id_list, mk_rec, name=None):
    """Extract given records by ID from metaknowledge collection into a new collection.

    Parameters
    ==========
    * id_list : array-like, list of IDs to extract.
    * mk_rec : Metaknowledge RecordCollection, to extract from
    * name : str, name of new collection. Optional.

    Returns
    ==========
    * Metaknowledge RecordCollection
    """
    new_rec = mk.RecordCollection(name=name)

    for doc_id in id_list:
        new_rec.add(mk_rec.getID(doc_id))

    return new_rec
# End extract_recs()


def rec_to_df(recs, extra_cols=None):
    """Deprecated function. Use `rc_to_df()` instead!

    Parameters
    ==========
    * rec : Metaknowledge RecordCollection
    * cols : list[str], column names to extract given as RIS field codes. If `None` extracts all relevant fields.

    Returns
    ==========
    * Pandas DataFrame
    """
    import warnings
    warnings.warn("Deprecated function `rec_to_df` - use `rc_to_df()` instead!", DeprecationWarning)
    return rc_to_df(recs, extra_cols)
# End rec_to_df()


def rc_to_df(recs, extra_cols=None):
    """Convert the given Metaknowledge RecordCollection to a Pandas DataFrame.

    Parameters
    ==========
    * rec : Metaknowledge RecordCollection
    * extra_cols : list[str], column names to extract given as RIS field codes. If `None` extracts all relevant fields.

    Returns
    ==========
    * Pandas DataFrame
    """
    if not extra_cols:
        extra_cols = ["AU", "SO", "DE", "DOI"]
    # df = pd.DataFrame(recs.forNLP(extraColumns=extra_cols))
    df = pd.DataFrame(recs.forNLP(extraColumns=extra_cols, removeCopyright=True, lower=False, removeNonWords=False))
    return df

    
