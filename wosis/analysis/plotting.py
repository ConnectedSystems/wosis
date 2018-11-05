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
        fig = func(*args, **kwargs)

        if save_plot_fn:
            if save_plot_fn.endswith('.png'):
                save_plot_fn = save_plot_fn.strip('.png')

            plt.tight_layout()
            fig.savefig(save_plot_fn+'.png', format='png', dpi=300)
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


def _prep_journal_records(search_results):
    journals = pd.DataFrame(search_results.forNLP(extraColumns=['SO']))  # SO is the field code for journal name

    # Some journal names have issue name in the title so remove these to allow better grouping
    journals.loc[:, 'SO'] = [x[0] for x in journals.loc[:, 'SO'].str.split('-')]

    return journals
# End _prep_journal_records()

@plot_saver
def plot_pubs_per_journal(search_results, top_n=10, annotate=False, print_stats=True):
    """Plot horizontal bar plot of publications for each journal in descending order.

    Parameters
    ==========
    * search_results : MetaKnowledge RecordCollection, of search results
    * top_n : int, number of journals to display (default: 10)
    * annotate : bool, annotate plot with values (default: False)
    * print_stats : bool, print out percentage of publications the results represent

    Returns
    ==========
    * matplotlib figure object

    """
    # SO is the field code for journal name
    journals = _prep_journal_records(search_results)
    pubs_by_journal = journals.groupby(('SO')).count()

    # rename column to 'count'
    pubs_by_journal.loc[:, 'count'] = pubs_by_journal.loc[:, 'year']
    pubs_by_journal.drop('year', inplace=True, axis=1)
    pubs_by_journal = pubs_by_journal.loc[:, 'count']

    pubs_by_journal = pubs_by_journal.sort_values(ascending=False)
    subtotal = pubs_by_journal.sort_values(ascending=False)[0:10].sum()
    plot_title = "Top {} Journals by Number of Publications".format(top_n)

    plt.tight_layout()
    ax = pubs_by_journal[0:top_n][::-1].plot(kind='barh', fontsize=12, title=plot_title, figsize=(10,6))
    ax.set_ylabel('')

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_width()), (p.get_width() + 0.01, p.get_y()), fontsize=12)

    if print_stats:
        print("{} out of {} ({:.2f}%)".format(subtotal, len(search_results), (subtotal/len(search_results)) * 100.0))

    return ax.get_figure()
# End plot_pubs_per_journal()


@plot_saver
def plot_pubs_across_time(search_results, top_n=10):
    """Plot publications across time by journal.

    Parameters
    ==========
    * search_results : MetaKnowledge RecordCollection, of search results
    * top_n : int, number of journals to display (default: 10)

    Returns
    ==========
    * matplotlib figure object
    """
    journals = _prep_journal_records(search_results)

    top_n_journals = journals.groupby(by=['SO']).count().sort_values('id', ascending=False).head(top_n)

    pubs_by_journal_year = journals.groupby(by=['SO', 'year'])
    pubs_for_journals = pubs_by_journal_year.count().sort_values('id', ascending=False).loc[top_n_journals.index, 'id']
    pubs_for_journals = pubs_for_journals.to_frame()
    pubs_for_journals.columns = ['Num. Publications']
    pubs_for_journals.index.name = 'Journal'

    pubs_across_time = pubs_for_journals.loc[:, 'Num. Publications'].unstack()
    pubs_across_time = pubs_across_time.fillna(0.0).transpose()
    pubs_across_time = pubs_across_time.sort_index()

    axes = pubs_across_time.plot(subplots=True, figsize=(6,8), layout=(10,1), sharey=True, legend=False)

    # Add legends (right hand side, outside of figure)
    [ax[0].legend([so], fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
     for ax, so in zip(axes, pubs_across_time.columns)]

    return axes[0][0].get_figure()
# End plot_pubs_across_time()
