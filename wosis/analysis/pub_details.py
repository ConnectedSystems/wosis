import pandas as pd


def citations(records):
    """Get citations
    Parameters
    ==========
    * records : Metaknowledge RecordCollection

    Returns
    ==========
    * Pandas DataFrame
    """
    paper_citations = pd.DataFrame(records.localCiteStats(pandasFriendly=True))
    paper_citations = paper_citations.set_index('Citations', drop=True)
    paper_citations.index.name = 'citation'

    return paper_citations.sort_values(by='Counts', ascending=False)
# End citations()
