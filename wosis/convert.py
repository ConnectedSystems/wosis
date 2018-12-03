import metaknowledge as mk

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
