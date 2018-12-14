import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import wraps, reduce
from matplotlib.ticker import MaxNLocator

from .search import get_unique_kw_titles
from wosis.convert import rc_to_df


def _truncate_string(string, delimiter="|", near=21):
    """The given string to the nearest delimiter, and add an ellipsis.

    Parameter
    =========
    * string : str, string to truncate
    * delimiter : str, the character used as a delimiter
    * near : int, find the nearest delimiter beyond this index position

    Returns
    =========
    * str, truncated string
    """
    last_bar_pos = string.rfind("|")
    if last_bar_pos > near:
        truncated_bar_pos = string.find("|", near)
        string = string[:truncated_bar_pos+1] + ' ...'
    return string
# End _truncate_string()


def plot_saver(func):
    """Decorator to enable all plotting functions to save figures.
    Figures are saved in `png` format at 300 dpi resolution.

    Added Parameter
    ================
    * save_plot_fn : str, indicate path to save figure
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        save_plot_fn = kwargs.pop('save_plot_fn', None)
        fig = func(*args, **kwargs)

        if save_plot_fn:
            if save_plot_fn.endswith('.png'):
                save_plot_fn = save_plot_fn.strip('.png')

            plt.tight_layout()
            fig.savefig(save_plot_fn + '.png', format='png', dpi=300)
        # End if
    # End wrapper()

    return wrapper
# End plot_saver()


@plot_saver
def plot_pub_trend(search_results, title=None, no_log_scale=False):
    """Plot publication trend across time.

    Will publication trend in log scale if large number of publications found.
    This can be avoided by setting `no_log_scale` to `True`.

    Parameters
    ==========
    * search_results : MetaKnowledge RecordCollection, of search results
    * title : str, title for plot
    * no_log_scale : bool, avoid log scale

    Returns
    ==========
    * matplotlib figure object
    """
    time_series = search_results.timeSeries(pandasMode=True)

    num_kwds = [len(ent['DE']) + len(ent['ID'])
                for ent in time_series['entry']]
    yearly = pd.DataFrame(
        {'year': time_series['year'], 'count': num_kwds}).groupby('year')
    num_pubs = yearly.count().sort_index()

    # Fill in the missing years
    min_year, max_year = num_pubs.index.min(), num_pubs.index.max()
    idx = pd.period_range(min_year, max_year, freq='Y')

    num_pubs = pd.DataFrame({'count': [num_pubs.loc[i, 'count']
                                       if i in num_pubs.index else 0 for i in idx.year]},
                            index=idx)

    fig, axes = plt.subplots(1)

    # Rotate x-axis labels if there is enough space
    rot = 45 if len(num_pubs.index) < 20 else 90

    pub_data = num_pubs.loc[:, 'count']
    num_text = "Total Number of Publications: {}".format(pub_data.sum())
    if title:
        title = title + '\n' + num_text
    else:
        title = num_text
    plt.suptitle(title, fontsize='22')

    tick_threshold = 11  # Hide every second year if number of years is above this

    if not no_log_scale:
        # use log scale if large values found
        log_form = True if max(pub_data) > 100 else False
    else:
        log_form = False
    # End if

    # force y-axis to use integer values
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    num_pubs.plot(kind='bar', figsize=(9, 6), ax=axes, rot=rot, logy=log_form, legend=False)

    if len(num_pubs.index) > tick_threshold:
        axes.set_xticks([i for i in range(0, len(num_pubs.index), 2)])
        axes.set_xticklabels([i.year for i in num_pubs.index[::2]])

    axes.set_xlabel("Year")
    ax_label = "Num. Publications"
    if log_form:
        ax_label += "\n(log scale)"

    axes.set_ylabel(ax_label)

    return fig
# End plot_pub_trend()


@plot_saver
def plot_kw_trend(search_results, title=None, no_log_scale=False):
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

    num_kwds = [len(ent['DE']) + len(ent['ID'])
                for ent in time_series['entry']]
    yearly = pd.DataFrame(
        {'year': time_series['year'], 'count': num_kwds}).groupby('year')
    num_pubs = yearly.count().sort_index()
    kw_trend = pd.DataFrame({'year': time_series['year'], 'count': num_kwds}).groupby(
        'year').sum().sort_index()
    avg_kw_per_pub = (kw_trend.loc[:, 'count'] / num_pubs.loc[:, 'count']).to_frame()

    # Fill in the missing years
    min_year, max_year = kw_trend.index.min(), kw_trend.index.max()
    idx = pd.period_range(min_year, max_year, freq='Y')
    avg_kw_per_pub = pd.DataFrame({'count': [avg_kw_per_pub.loc[i, 'count']
                                             if i in avg_kw_per_pub.index else 0 for i in idx.year]}, index=idx)
    num_pubs = pd.DataFrame({'count': [num_pubs.loc[i, 'count'] if i in num_pubs.index else 0 for i in idx.year]},
                            index=idx)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Rotate x-axis labels if there is enough space
    rot = 45 if len(kw_trend) < 20 else 90

    pub_data = num_pubs.loc[:, 'count']
    num_text = "Total Number of Publications: {}".format(pub_data.sum())
    if title:
        title = title + '\n' + num_text
    else:
        title = num_text
    plt.suptitle(title, fontsize='22')

    avg_kw_per_pub.plot(kind='bar', figsize=(
        18, 6), rot=rot, ax=ax1, legend=False)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average Num. Keywords")

    tick_threshold = 11  # Hide every second year if number of years is above this
    if len(avg_kw_per_pub.index) > tick_threshold:
        ax1.set_xticks([i for i in range(0, len(avg_kw_per_pub.index), 2)])
        ax1.set_xticklabels([i.year for i in avg_kw_per_pub.index[::2]])

    if not no_log_scale:
        # use log scale if large values found
        log_form = True if max(pub_data) > 100 else False
    else:
        log_form = False
    # End if

    # force y-axis to use integer values
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    num_pubs.plot(kind='bar', ax=ax2, rot=rot, logy=log_form, legend=False)

    if len(num_pubs.index) > tick_threshold:
        ax2.set_xticks([i for i in range(0, len(num_pubs.index), 2)])
        ax2.set_xticklabels([i.year for i in num_pubs.index[::2]])

    ax2.set_xlabel("Year")
    ax_label = "Num. Publications"
    if log_form:
        ax_label += "\n(log scale)"

    ax2.set_ylabel(ax_label)

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
    ptitle = top_title + \
        "\n{} unique titles out of {}".format(num_titles, len(corpora))
    pubs_per_kw = pd.DataFrame(list(summary.items()), index=list(
        summary.keys()), columns=['Keyword', 'Count'])
    pubs_per_kw.sort_values(by='Count', inplace=True)

    ax = pubs_per_kw.plot(kind='bar', title=ptitle, figsize=(8, 6))

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_height()), (p.get_x() +
                                                      0.01, p.get_height() * 1.015), fontsize=14)

    # force y-axis to use integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    return ax.get_figure()
# End plot_pub_per_kw()


def _prep_journal_records(search_results):
    # SO is the field code for journal name
    journals = pd.DataFrame(search_results.forNLP(extraColumns=['SO']))

    # Some journal names have issue name in the title so remove these to allow better grouping
    journals.loc[:, 'SO'] = [x[0]
                             for x in journals.loc[:, 'SO'].str.split('-')]

    return journals
# End _prep_journal_records()


@plot_saver
def plot_pubs_per_journal(search_results, top_n=10, annotate=False, show_stats=True):
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

    if show_stats:
        plot_title += "\n{} out of {} ({:.2f}%)".format(subtotal, len(
            search_results), (subtotal / len(search_results)) * 100.0)

    plt.tight_layout()
    ax = pubs_by_journal[0:top_n][::-
                                  1].plot(kind='barh', fontsize=12, title=plot_title, figsize=(10, 6))
    ax.set_ylabel('')

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_width()),
                        (p.get_width() + 0.01, p.get_y()), fontsize=12)

    return ax.get_figure()
# End plot_pubs_per_journal()


@plot_saver
def plot_journal_pub_trend(search_results, top_n=10):
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

    top_n_journals = journals.groupby(by=['SO']).count(
    ).sort_values('id', ascending=False).head(top_n)

    pubs_by_journal_year = journals.groupby(by=['SO', 'year'])
    pubs_for_journals = pubs_by_journal_year.count().sort_values(
        'id', ascending=False).loc[top_n_journals.index, 'id']
    pubs_for_journals = pubs_for_journals.to_frame()
    pubs_for_journals.columns = ['Num. Publications']
    pubs_for_journals.index.name = 'Journal'

    pubs_across_time = pubs_for_journals.loc[:, 'Num. Publications'].unstack()
    pubs_across_time = pubs_across_time.fillna(0.0).transpose()
    pubs_across_time = pubs_across_time.sort_index()

    axes = pubs_across_time.plot(subplots=True, figsize=(
        6, 8), layout=(10, 1), sharey=True, legend=False)

    # Add legends (right hand side, outside of figure)
    [ax[0].legend([so], fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
     for ax, so in zip(axes, pubs_across_time.columns)]

    return axes[0][0].get_figure()
# End plot_journal_pub_trend()


@plot_saver
def plot_criteria_trend(corpora_df, threshold=3):
    """Plot criteria membership across time.

    Parameters
    ==========
    * corpora_df : Pandas DataFrame, of records from wosis.analysis.search.collate_keyword_criteria_matches
    * threshold : int, plot number of papers that are members of at least this number of criterias (default: 3)

    Returns
    ==========
    * matplotlib figure object

    See Also
    ==========
    * wosis.analysis.search.collate_keyword_criteria_matches
    """
    if 'num_criteria_match' not in corpora_df:
        err_msg = 'DataFrame must have "num_criteria_match" column.\n\
See `wosis.analysis.search.collate_keyword_criteria_matches`'
        raise KeyError(err_msg)
    # End if

    match_col = corpora_df['num_criteria_match']
    grp_count = corpora_df.loc[match_col >= threshold, :].groupby(
        'year')['num_criteria_match'].count()
    current_palette = sns.color_palette()

    ax = grp_count.plot(kind='bar', color=current_palette[0],
                        title='Papers with Keywords\nin {} or More Criteria'.format(threshold))

    return ax.get_figure()
# End plot_criteria_trend()


@plot_saver
def plot_topic_trend(topic_summaries, total_rc=None, title='Topic Trend'):
    """Plot the trends of topics over time.

    Parameters
    ==========
    * topic_summaries : list[tuple], of topics based on the output of `wosis.analysis.keyword_matches()`
    * total_rc : RecordCollection, collection used to calculate topic proportion relative to the corpora.

    Returns
    ==========
    * Matplotlib plot object

    See Also
    ==========
    * wosis.analysis.keyword_matches()
    """
    if total_rc:
        rc = pd.DataFrame(total_rc.timeSeries('year'))
        rc = rc.set_index('year', drop=True)
        rc = rc['count']
        y_label = 'Perc. of Corpora (%)'
        mod = 100.0
    else:
        rc = 1
        mod = 1
        y_label = 'Num. Publications'

    ax = None
    alpha_val = 0.7 if len(topic_summaries) > 1 else 1.0
    for topic in topic_summaries:
        rcs, summary = topic
        if isinstance(rcs, dict):
            rcs = reduce(lambda x, y: x + y, rcs.values())

        label = " | ".join(summary.keys())
        label = _truncate_string(label, "|", 21)

        df = pd.DataFrame(rcs.timeSeries('year'))
        df = df.set_index('year', drop=True)

        ax = ((df['count'] / rc) * mod).plot(legend=True, ax=ax, label=label, style='-o', alpha=alpha_val)
    # End for

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # force y-axis to use integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(y_label)

    return ax.get_figure()
