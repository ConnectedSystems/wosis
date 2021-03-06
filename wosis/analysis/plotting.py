import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

import seaborn as sns

from functools import wraps, reduce

from .search import get_unique_kw_titles
from wosis.convert import rc_to_df


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
        ret = kwargs.pop('return_fig', None)

        axes = func(*args, **kwargs)
        if isinstance(axes, list):
            fig = axes[0].get_figure()
        else:
            fig = axes.get_figure()
        # End if

        plt.tight_layout()

        if save_plot_fn:
            if save_plot_fn.endswith('.png'):
                save_plot_fn = save_plot_fn.replace('.png', '')

            fig.savefig(save_plot_fn + '.png', format='png', 
                        dpi=300, bbox_inches='tight')
        # End if

        if ret:
            return axes
    # End wrapper()

    return wrapper
# End plot_saver()


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


def _set_title(title, num_text):
    """Determine title text and modified position."""
    y_pos = 0.98
    if title:
        title = title + '\n' + num_text
        y_pos = 1.08
    else:
        title = num_text
        y_pos = 1.01

    return title, y_pos
# End _set_title()


@plot_saver
def plot_pub_trend(search_results, title=None, no_log_scale=False, ax=None):
    """Plot publication trend across time.

    Will publication trend in log scale if large number of publications found.
    This can be avoided by setting `no_log_scale` to `True`.

    Parameters
    ==========
    * search_results : MetaKnowledge RecordCollection, of search results
    * title : str, title for plot
    * no_log_scale : bool, avoid log scale
    * ax : Axis Object or None, matplotlib axis object to add to
    * kwargs: additional args passed to matplotlib

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

    if not ax:
        fig, ax = plt.subplots(1)

    # Rotate x-axis labels if there is enough space
    rot = 45 if len(num_pubs.index) < 20 else 90

    pub_data = num_pubs.loc[:, 'count']
    num_text = "Total Number of Publications: {}".format(pub_data.sum())
    
    title, y_pos = _set_title(title, num_text)
    plt.suptitle(title, fontsize='22', y=y_pos)

    tick_threshold = 11  # Hide every second year if number of years is above this

    if not no_log_scale:
        # use log scale if large values found
        log_form = True if max(pub_data) > 100 else False
    else:
        log_form = False
    # End if

    # force y-axis to use integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    num_pubs.plot(kind='bar', figsize=(9, 6), ax=ax, rot=rot, logy=log_form, legend=False)

    if len(num_pubs.index) > tick_threshold:
        ax.set_xticks([i for i in range(0, len(num_pubs.index), 2)])
        ax.set_xticklabels([i.year for i in num_pubs.index[::2]])

    ax.set_xlabel("Year")
    ax_label = "Num. Publications"
    if log_form:
        ax_label += "\n(log scale)"

    ax.set_ylabel(ax_label)

    return ax
# End plot_pub_trend()


@plot_saver
def plot_kw_trend(search_results, title=None, no_log_scale=False, ax=None):
    """Plot keyword trends across time.

    Parameters
    ==========
    * search_results : MetaKnowledge RecordCollection, of search results
    * ax : List[Axis Object] or None, matplotlib axis object to add to

    See Also
    ==========
    * wosis.analysis.search.find_keywords()

    Returns
    ==========
    * matplotlib figure object
    """
    time_series = search_results.timeSeries(pandasMode=True)

    if len(time_series['entry'][0]['DE']) == 1:
        num_kwds = [len(ent['DE'][0].split())  
                    for ent in time_series['entry']]
    else:
        num_kwds = [len(ent['DE'])
                    for ent in time_series['entry']]
    # End if
    
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

    if not ax:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        ax1, ax2 = ax

    # Rotate x-axis labels if there is enough space
    rot = 45 if len(kw_trend) < 20 else 90

    pub_data = num_pubs.loc[:, 'count']
    num_text = "Total Number of Publications: {}".format(pub_data.sum())

    title, y_pos = _set_title(title, num_text)
    plt.suptitle(title, fontsize='22', y=y_pos)

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
    ax2 = num_pubs.plot(kind='bar', ax=ax2, rot=rot, logy=log_form, legend=False)

    if len(num_pubs.index) > tick_threshold:
        ax2.set_xticks([i for i in range(0, len(num_pubs.index), 2)])
        ax2.set_xticklabels([i.year for i in num_pubs.index[::2]])

    ax2.set_xlabel("Year")
    ax_label = "Num. Publications"
    if log_form:
        ax_label += "\n(log scale)"

    ax2.set_ylabel(ax_label)

    return fig.axes
# End plot_kw_trend()


@plot_saver
def plot_pub_per_kw(kw_matches, corpora, kw_category, annotate=False, ax=None):
    """Plot publications per keyword.

    Parameters
    ==========
    * kw_matches : KeywordMatch object
    * corpora : Metaknowledge Collection, representing corpora
    * kw_category : str, text indicating keyword category for use in plot title
    * annotate : bool, display number of records in plot
    * ax : Axis Object or None, matplotlib axis object to add to

    Example
    ==========
    ```python
    # where RC is some RecordCollection
    keywords = set(["software practice", "software development", "software engineering", 
                "best practice", "modeling practice"])
    matches = wosis.keyword_matches(RC, keywords, 95.0)
    wos_plot.plot_pub_per_kw(matches, RC, 'Practices')
    ```

    See Also
    ==========
    * wosis.analysis.search.keyword_matches()
    * wosis.analysis.search.get_unique_kw_titles()

    Returns
    ==========
    * matplotlib figure object
    """
    ind_recs, summary = kw_matches.recs, kw_matches.summary
    unique_titles = get_unique_kw_titles(ind_recs)
    num_titles = len(unique_titles)
    top_title = "Num. Publications per {} Keyword".format(kw_category.title())
    top_title = ' '.join(top_title.split())
    ptitle = top_title + \
        "\n{} unique titles out of {}".format(num_titles, len(corpora))
    pubs_per_kw = pd.DataFrame(list(summary.items()), index=list(
        summary.keys()), columns=['Keyword', 'Count'])
    pubs_per_kw.sort_values(by='Count', inplace=True)

    ax = pubs_per_kw.plot(kind='bar', title=ptitle, figsize=(8, 6), ax=ax)

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_height()), (p.get_x() +
                                                      0.01, p.get_height() * 1.015), fontsize=14)

    # force y-axis to use integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    return ax
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
def plot_pubs_per_journal(search_results, top_n=10, annotate=False, show_stats=True, title=None, ax=None):
    """Plot horizontal bar plot of publications for each journal in descending order.

    Parameters
    ==========
    * search_results : MetaKnowledge RecordCollection, of search results
    * top_n : int, number of journals to display (default: 10)
    * annotate : bool, annotate plot with values (default: False)
    * print_stats : bool, print out percentage of publications the results represent
    * ax : Axis Object or None, matplotlib axis object to add to

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

    if title is None:
        plot_title = "Top {} Journals by Number of Publications".format(top_n)
    else:
        plot_title = title

    if show_stats:
        plot_title += "\n{} out of {} ({:.2f}%)".format(subtotal, len(
            search_results), (subtotal / len(search_results)) * 100.0)

    ax = pubs_by_journal[0:top_n][::-1]\
        .plot(kind='barh', 
              fontsize=12, 
              title=plot_title, 
              figsize=(12, 6), 
              color='#2b7bba',  # match seaborn blue
              ax=ax
            )
    ax.set_ylabel('')

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_width()),
                        (p.get_width() + 0.01, p.get_y()), fontsize=12)

    return ax
# End plot_pubs_per_journal()


@plot_saver
def plot_journal_pub_trend(search_results, title='Journal Publication Trend', top_n=10):
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

    # Reorder based on total publications
    pubs_across_time = pubs_across_time[top_n_journals.index]

    axes = pubs_across_time.plot(subplots=True, figsize=(
        12, 10), layout=(top_n, 1), sharey=True, legend=False)

    plt.suptitle(title, fontsize='22', y=1.02)

    # force x-axis to use integer values (years)
    for ax in axes:
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add legends (right hand side, outside of figure)
    [ax[0].legend([so], fontsize=10, loc='center left', bbox_to_anchor=(1.0, 0.5))
     for ax, so in zip(axes, pubs_across_time.columns)]

    return axes[0][0]
# End plot_journal_pub_trend()


@plot_saver
def plot_criteria_trend(corpora_df, threshold=3, ax=None):
    """Plot criteria membership across time.

    Parameters
    ==========
    * corpora_df : Pandas DataFrame, of records from wosis.analysis.search.collate_keyword_criteria_matches
    * threshold : int, plot number of papers that are members of at least this number of criterias (default: 3)
    * ax : Axis Object or None, matplotlib axis object to add to

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
                        title='Papers with Keywords\nin {} or More Criteria'.format(threshold),
                        ax=ax)

    return ax
# End plot_criteria_trend()


@plot_saver
def plot_topic_trend(topics, total_rc=None, title='Topic Trend', ax=None):
    """Plot the trends of topics over time.

    Parameters
    ==========
    * topics : list[KeywordMatch], KeywordMatch object of topics based on the output of `wosis.analysis.keyword_matches()`
    * total_rc : RecordCollection or None, collection used to calculate topic proportion relative to the corpora.
                 If `None`, plots number of publications. Defaults to None.
    * title : str, title for plot
    * ax : Axis Object or None, matplotlib axis object to add to

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
        y_label = 'Percentage of Corpora (%)'
        mod = 100.0
    else:
        rc = 1
        mod = 1
        y_label = 'Num. Publications'

    alpha_val = 0.7 if len(topics) > 1 else 1.0

    if not ax:
        fig, ax = plt.subplots(figsize=(12,6))

    plt.title(title)
    for kwm in topics:
        if hasattr(kwm, 'recs'):
            rcs = kwm.combine_recs()
            label = " | ".join(kwm.recs.keys())
        else:
            # assume it is a metaknowledge object
            rcs = kwm
            label = rcs.name
        
        label = _truncate_string(label, "|", 11)

        df = pd.DataFrame(rcs.timeSeries('year'))
        df = df.set_index('year', drop=True)

        label = label + "\n({} publications)".format(df['count'].sum())

        ax = ((df['count'] / rc) * mod).plot(legend=False, ax=ax, label=label, style='-o', alpha=alpha_val)
    # End for

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # force axis to use integer values
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(y_label)

    return ax
# End plot_topic_trend()


@plot_saver
def plot_citations(citation_df, top_n=10, plot_title='Citations', annotate=True, ax=None):
    """Plot citations per top `n` papers.

    Parameters
    ==========
    * citation_df : DataFrame, with citations column
    * top_n : int, top `n` papers to show
    * annotate : bool, show number of citations. Defaults to True.
    * ax : Axis Object or None, matplotlib axis object to add to

    Returns
    ==========
    * Matplotlib plot object
    """
    citation_df.index = citation_df['title'].str[0:25] + "..."
    ax = citation_df['citations'][:top_n][::-1].plot(kind='barh', 
                                                     color='blue', 
                                                     title=plot_title, 
                                                     fontsize=10, 
                                                     figsize=(10,6),
                                                     ax=ax)

    if annotate:
        # Annotate number above bar
        for p in ax.patches:
            ax.annotate("{}".format(p.get_width()),
                        (p.get_width() + 0.01, p.get_y()), fontsize=12)

    return ax
# End plot_citations()
