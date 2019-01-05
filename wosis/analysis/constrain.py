import pandas as pd

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from wosis import rc_to_df
from wosis.TopicResult import TopicResult

import warnings
from zipfile import BadZipfile

__all__ = ['find_topics', 'remove_by_journals', 'remove_by_title', 'remove_empty_DOIs']

# We lemmatize and stem words to homogenize similar content as much as possible
lemmer = WordNetLemmatizer().lemmatize
stemmer = SnowballStemmer('english').stem

def _homogenize(word):
    return stemmer(lemmer(word))


def find_topics(corpora, model_type='NMF', num_topics=10, num_features=1000, verbose=True):
    """Using one of several approaches, try to identify topics to help constrain search space.

    Parameters
    ==========
    * corpora_df : Pandas DataFrame, Corpora derived from Metaknowledge RecordCollection
    * model_type : str, name of topic modeling approach
    * num_topics : int, attempt to sort documents into this number of topics
    * num_features : int, essentially the maximum number of words to consider per corpus
    * verbose : bool, print out information or not. Default to True

    Returns
    ==========
    * tuple, topic model and feature names
    """
    if 'metaknowledge' in str(type(corpora)).lower():
        corpora_df = rc_to_df(corpora)
        try:
            filtered_corpora_df = pd.DataFrame(corpora.forNLP(extraColumns=["AU", "SO", "DE"],
                                                            stemmer=_homogenize))
        except BadZipfile:
            warnings.warn("Could not stem/lemmatize content - set up NLTK WordNet data first!")
            filtered_corpora_df = pd.DataFrame(corpora.forNLP(extraColumns=["AU", "SO", "DE"]))
    elif 'dataframe' in str(type(corpora)).lower():
        corpora_df = corpora

    combined_kws = corpora_df['DE'].str.split("|").tolist()
    corpora_df.loc[:, "kws"] = [" ".join(i) for i in combined_kws]
    docs = corpora_df['title'] + corpora_df['abstract'] + corpora_df["kws"]

    if model_type is 'NMF':
        mod, trans, names = NMF_cluster(docs, num_topics, num_features)
    elif model_type is 'LDA':
        mod, trans, names = LDA_cluster(docs, num_topics, num_features)
    else:
        raise ValueError("Unknown or unimplemented topic modelling approach!")
    # End if

    res = TopicResult(mod, trans, names, corpora_df)

    if verbose:
        res.display_topics()

    return res

# End find_topics()


def cluster_topics(approach, docs, num_features, stop_words='english', verbose=True):

    vec = approach(max_df=0.95, min_df=2,
                   max_features=num_features, stop_words=stop_words)
    trans = vec.fit_transform(docs)
    feature_names = vec.get_feature_names()

    return trans, feature_names
# End cluster_topics


def NMF_cluster(docs, num_topics, num_features, stop_words='english', verbose=True):
    trans, names = cluster_topics(
        TfidfVectorizer, docs, num_features, stop_words, verbose)

    # Run NMF
    nmf = NMF(n_components=num_topics, random_state=1, alpha=.1,
              l1_ratio=.5, init='nndsvd', max_iter=50).fit(trans)

    return nmf, trans, names
# End NMF_cluster()


def LDA_cluster(docs, num_topics, num_features, stop_words='english', verbose=True):
    trans, names = cluster_topics(
        CountVectorizer, docs, num_features, stop_words, verbose)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50,
                                    learning_method='online', learning_offset=50., random_state=0).fit(trans)

    return lda, trans, names
# End LDA_cluster()


def remove_by_journals(corpora, unrelated_journals, verbose=True):
    # TODO : Apply this to the RecordCollection instead of DF
    for unrelated in unrelated_journals:
        count_removed = corpora.loc[corpora['SO'].str.contains(
            unrelated), 'id'].count()
        if verbose:
            print("{}: {}".format(unrelated, count_removed))
        corpora = corpora.drop(corpora.loc[corpora['SO'].str.contains(unrelated)].index,
                               axis=0)
    return corpora
# End remove_by_journals()


def remove_by_title(corpora, unrelated_titles, verbose=True):
    # TODO : Apply this to the RecordCollection instead of D
    for unrelated in unrelated_titles:
        count_removed = corpora.loc[corpora['title'].str.contains(
            unrelated), 'id'].count()

        if verbose:
            print("{}: {}".format(unrelated, count_removed))

        corpora = corpora.drop(
            corpora.loc[corpora['title'].str.contains(unrelated)].index, axis=0)

    return corpora
# End remove_by_title()


def remove_empty_DOIs(corpora, return_removed=False, verbose=True):
    """Remove records with no associated DOI from DataFrame.

    Parameters
    ==========
    * corpora : Pandas DataFrame
    * return_removed : bool,
    * verbose : bool, print out information during removal process. Defaults to True.

    Returns
    ==========
    * tuple[Pandas DataFrame], Filtered DataFrame and DataFrame of removed records
    """
    to_be_removed = corpora.loc[corpora['DOI'] == '', :]
    if verbose:
        count_empty = to_be_removed['DOI'].count()
        print("Removing {} records with no DOIs".format(count_empty))

    filtered = corpora.loc[corpora['DOI'] != '', :]

    return filtered, to_be_removed
# End remove_empty_DOIs()
