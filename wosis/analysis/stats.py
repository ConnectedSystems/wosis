__all__ = ['calc_average_citations']

def calc_average_citations(in_df, max_year=None, sort=False):
    """Calculate the average citations since year of publication.
    
    If no `max_year` is specified, calculate using the latest 
    year in given dataset.

    Example
    ==========
    citations = wosis.get_num_citations(corpora_rc, WOS_CONFIG)
    avg_citations = wosis.calc_average_citations(citations)

    See also
    ==========
    * `wosis.get_num_citations()`

    Parameters
    ==========
    * in_df : Pandas DataFrame
    * max_year: int or None, year to calculate average citations from.

    Returns
    ==========
    * A copy of the DataFrame with 'Avg. Citations' column
    """
    assert hasattr(in_df, 'citations'), \
        'DataFrame has to have `citation` column. Use `get_num_citations()` first'
    out_df = in_df.copy()
    
    max_year_in_data = out_df.year.max()
    
    if max_year is not None:
        max_year = int(max_year)
        assert max_year_in_data <= max_year, \
            "Given max_year must be later than any year found in dataset."
    else:
        max_year = out_df.year.max()
    out_df.loc[:, 'Avg. Citations'] = (out_df.citations / ((max_year - out_df.year) + 1)).astype(float).round(2)

    if sort:
        out_df = out_df.sort_values('Avg. Citations', ascending=False)

    return out_df
# End calc_average_citations()

