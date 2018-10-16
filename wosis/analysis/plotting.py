import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from .search import get_unique_kw_titles

sns.set_style('darkgrid')
sns.set_context('paper', font_scale=1.4)

plt.tight_layout()


def plot_saver(func):
    """Decorator to enable all plotting functions to save figures"""

    def wrapper(*args, **kwargs):
        save_plot_fn = kwargs.pop('save_plot_fn', None)
        plot = func(*args, **kwargs)

        if save_plot_fn:
            if save_plot_fn.endswith('.png'):
                save_plot_fn = save_plot_fn.strip('.png')
            plot.savefig(save_plot_fn+'.png', format='png', dpi=300)
        # End if
    # End wrapper()

    return wrapper
# End plot_saver()


@plot_saver
def plot_kw_trend(search_results):
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
    trend = {}
    num_pubs = {}
    for rec in search_results:
        year = rec.get('PY')  # Get publication year
        if year not in trend:
            trend[year] = 0
            num_pubs[year] = 0

        num_all_kwds = len(rec.get('DE')) + len(rec.get('ID'))
        trend[year] += num_all_kwds
        num_pubs[year] += 1

    avg_kw_per_pub = np.array(list(trend.values())) / np.array(list(num_pubs.values()))

    df = pd.DataFrame({'Num. Keywords': avg_kw_per_pub}, index=trend.keys()).sort_index()

    fig, (ax1, ax2) = plt.subplots(1,2)
    rot = 45 if len(trend) < 20 else 90

    df.plot(kind='bar', figsize=(18, 6), rot=rot, title="Average Number of Keywords per Publication", ax=ax1, legend=False)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average Num. Keywords");

    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # force y-axis to use integer values
    pub_data = num_pubs.values()
    num_pubs = pd.DataFrame({'Num. Publications': list(pub_data)}, index=num_pubs.keys()).sort_index()
    num_pub_title = "Number of Publications\nTotal: {}".format(sum(pub_data))

    log_form = True if max(pub_data) > 100 else False

    num_pubs.plot(kind='bar', ax=ax2, rot=rot, logy=log_form, legend=False, title=num_pub_title)
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
    * get_unique_kw_titles

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

    ax = pubs_per_kw.plot(kind='bar', title=ptitle, rot=45, figsize=(8,6))

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_height()), (p.get_x() + 0.01, p.get_height() * 1.015))

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # force y-axis to use integer values

    plt.tight_layout()
    return ax.get_figure()
# End plot_pub_per_kw()
