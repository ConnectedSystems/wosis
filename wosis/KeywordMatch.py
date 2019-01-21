from functools import reduce

class KeywordMatch(object):
    """Holds keyword match information"""

    def __init__(self, recs):
        """
        Parameters
        ==========
        * recs : dict, {'keyword': MetaKnowledge RecordCollection}
        """

        self.recs = recs
        self.summary = {rc.name: len(rc) for rc in recs.values()}
    # End __init__()

    def combine_recs(self):
        """Combine all records into one"""
        return reduce(lambda x, y: x + y, self.recs.values())

    @property
    def name(self):
        names = " | ".join([rc.name for rc in self.recs.values()])

    def __len__(self):
        return sum([len(rc) for rc in self.recs.values()])

    def __add__(self, other):
        if hasattr(other, 'recs'):
            return self.combine_recs() + other.combine_recs()
        
        if hasattr(other, 'name'):
            # Assume other is a metaknowledge collection
            return self.combine_recs() + other
        
        raise ValueError("Cannot add {} to {}".format(type(self), type(other)))


