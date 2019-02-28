__all__ = ['calc_average_citations']

def calc_average_citations(in_df, sort=False):
    """Calculate the average citations since year of publication.

    Example
    ==========
    citations = wosis.get_num_citations(corpora_rc, WOS_CONFIG)
    avg_citations = wosis.calc_average_citations(citations)

    See also
    ==========
    * `wosis.get_num_citations()`

    Parameters
    ==========
    * df : Pandas DataFrame

    Returns
    ==========
    * A copy of the DataFrame with 'Avg. Citations' column
    """
    assert hasattr(in_df, 'citations'), \
        'DataFrame has to have `citation` column. Use `get_num_citations()` first'
    out_df = in_df.copy()
    max_year = out_df.year.max()
    out_df.loc[:, 'Avg. Citations'] = (out_df.citations / ((max_year - out_df.year) + 1)).round(2)

    if sort:
        out_df = out_df.sort_values('Avg. Citations', ascending=False)

    return out_df
# End calc_average_citations()

