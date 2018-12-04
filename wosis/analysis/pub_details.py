import pandas as pd


def pub_citations(records):
    """Get citations for each publication.

    Parameters
    ==========
    * records : Metaknowledge RecordCollection

    Returns
    ==========
    * Pandas DataFrame
    """
    paper_citations = pd.DataFrame(records.localCiteStats(pandasFriendly=True))
    paper_citations = paper_citations.set_index('Citations', drop=True)
    paper_citations.index.name = 'Publication'
    paper_citations = paper_citations.rename(columns={'Counts': 'Citations'})

    return paper_citations.sort_values(by='Citations', ascending=False)
# End citations()


def author_citations(records):
    """Get citations by author.

    WARNING: Be careful interpreting this - author names are grouped by surname so it is misleading!

    Parameters
    ==========
    * records : Metaknowledge RecordCollection

    Returns
    ==========
    * Pandas DataFrame
    """
    # Authors with most citations (careful interpreting this - author names are grouped by surname so it is misleading)
    author_citations = pd.DataFrame(records.localCiteStats(pandasFriendly=True, keyType='author'))
    author_citations = author_citations.set_index('Citations', drop=True)
    author_citations.index.name = 'Publication'
    author_citations = author_citations.rename(columns={'Counts': 'Citations'})

    return author_citations.sort_values(by='Citations', ascending=False)
# End author_citations()


def link_to_pub(records):
    if 'metaknowledge' in str(type(records)):
        recs = records.forNLP(extraColumns=["AU", "SO", "DE", 'DOI'], lower=False, removeNonWords=False)
        df = pd.DataFrame(recs)
    elif 'DataFrame' in str(type(records)):
        df = records
    # End if

    df.loc[df['DOI'] != '', 'DOI link'] = "https://dx.doi.org/" + df.loc[df['DOI'] != '', 'DOI'].astype(str)

    return df
# End link_to_pub()
