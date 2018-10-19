import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from .search import get_unique_kw_titles


def plot_saver(func):
    """Decorator to enable all plotting functions to save figures.
    Figures are saved in `png` format at 300 dpi resolution.

    Added Parameter
    ================
    * save_plot_fn : str, indicate path to save figure
    """

    def wrapper(*args, **kwargs):
        save_plot_fn = kwargs.pop('save_plot_fn', None)
        plot = func(*args, **kwargs)

        if save_plot_fn:
            if save_plot_fn.endswith('.png'):
                save_plot_fn = save_plot_fn.strip('.png')

            plt.tight_layout()
            plot.savefig(save_plot_fn+'.png', format='png', dpi=300)
        # End if
    # End wrapper()

    return wrapper
# End plot_saver()


@plot_saver
def plot_kw_trend(search_results, title=None):
    """Plot keyword trends across time.

    Parameters
    ==========
    * search_results : MetaKnowledge RecordCollection, of search results

    See Also
    ==========
    * wosis.analysis.search.search_records()

    Returns
    ==========
    * matplotlib figure object
    """
    time_series = search_results.timeSeries(pandasMode=True)

    num_kwds = [len(ent['DE']) + len(ent['ID']) for ent in time_series['entry']]
    yearly = pd.DataFrame({'year': time_series['year'], 'count': num_kwds}).groupby('year')
    num_pubs = yearly.count().sort_index()
    kw_trend = pd.DataFrame({'year': time_series['year'], 'count': num_kwds}).groupby('year').sum().sort_index()
    avg_kw_per_pub = (kw_trend.loc[:, 'count'] / num_pubs.loc[:, 'count']).to_frame()

    # Fill in the missing years
    min_year, max_year = kw_trend.index.min(), kw_trend.index.max()
    idx = pd.period_range(min_year, max_year, freq='Y')
    avg_kw_per_pub = pd.DataFrame({'count': [avg_kw_per_pub.loc[i, 'count']
                                             if i in avg_kw_per_pub.index else 0 for i in idx.year]}, index=idx)
    num_pubs = pd.DataFrame({'count': [num_pubs.loc[i, 'count'] if i in num_pubs.index else 0 for i in idx.year]},
                            index=idx)

    fig, (ax1, ax2) = plt.subplots(1,2)

    # Rotate x-axis labels if there is enough space
    rot = 45 if len(kw_trend) < 20 else 90
    fig.subplots_adjust(top=0.3)  # make space for top title

    pub_data = num_pubs.loc[:, 'count']
    num_text = "Total Number of Publications: {}".format(pub_data.sum())
    if title:
        title = title+'\n'+num_text
    else:
        title = num_text
    plt.suptitle(title, fontsize='22')

    avg_kw_per_pub.plot(kind='bar', figsize=(18, 6), rot=rot, ax=ax1, legend=False)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average Num. Keywords");

    log_form = True if max(pub_data) > 100 else False  # use log scale if large values found

    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # force y-axis to use integer values
    num_pubs.plot(kind='bar', ax=ax2, rot=rot, logy=log_form, legend=False)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Num. Publications")

    return fig
# End plot_kw_trend()


@plot_saver
def plot_pub_per_kw(ind_recs, summary, corpora, kw_category, annotate=False):
    """Plot publications per keyword.

    Parameters
    ==========
    * ind_recs : dict, of keywords and matching publication records
    * summary : dict, of keywords and matching number of publications
    * corpora : Metaknowledge Collection, representing corpora
    * kw_category : str, text indicating keyword category for use in plot title

    See Also
    ==========
    * wosis.analysis.search.keyword_matches()
    * wosis.analysis.search.get_unique_kw_titles()

    Returns
    ==========
    * matplotlib figure object
    """
    unique_titles = get_unique_kw_titles(ind_recs)
    num_titles = len(unique_titles)
    top_title = "Num. Publications per {} Keyword".format(kw_category.title())
    top_title = ' '.join(top_title.split())
    ptitle = top_title + "\n{} unique titles out of {}".format(num_titles, len(corpora))
    pubs_per_kw = pd.DataFrame(list(summary.items()), index=list(summary.keys()), columns=['Keyword', 'Count'])
    pubs_per_kw.sort_values(by='Count', inplace=True)

    ax = pubs_per_kw.plot(kind='bar', title=ptitle, figsize=(8,6))

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_height()), (p.get_x() + 0.01, p.get_height() * 1.015), fontsize=14)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # force y-axis to use integer values

    plt.tight_layout()
    return ax.get_figure()
# End plot_pub_per_kw()
