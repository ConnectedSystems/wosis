import pandas as pd


def pub_citations(records):
    """Get number of times a reference is used in the corpora.

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


def citations_of(records):
    """Get number of times each paper is cited within the corpora.

    Parameters
    ==========
    * records : Metaknowledge RecordCollection

    Returns
    ==========
    * Pandas DataFrame
    """
    citations = {}
    for pub_rec in records:
        citations[pub_rec.get('title')] = len(records.localCitesOf(pub_rec.createCitation()))

    res = pd.DataFrame({'Citations': list(citations.values())}, index=list(citations.keys()))
    return res.sort_values(by='Citations', ascending=False)


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
    """Adds link to publication.

    Parameters
    ==========
    * records : Metaknowledge RecordCollection or DataFrame

    Returns
    ==========
    * Pandas DataFrame with additional column ('DOI link')
    """
    if 'metaknowledge' in str(type(records)).lower():
        recs = records.forNLP(extraColumns=["AU", "SO", "DE", 'DOI'], lower=False, removeNonWords=False)
        df = pd.DataFrame(recs)
    elif 'dataframe' in str(type(records)).lower():
        df = records
    # End if

    df.loc[df['DOI'] != '', 'DOI link'] = "https://dx.doi.org/" + df.loc[df['DOI'] != '', 'DOI'].astype(str)

    return df
# End link_to_pub()
