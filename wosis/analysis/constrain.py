import pandas as pd

from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def display_topics(model, feature_names, num_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        match = " ".join([feature_names[i]
                          for i in topic.argsort()[:-num_top_words - 1:-1]])
        print("Topic {}: {}".format(topic_idx + 1, match))
# End display_topics()


def find_topics(corpora_df, model_type='NMF', num_topics=10, num_features=1000, verbose=True):
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
    combined_kws = corpora_df['DE'].str.split("|").tolist()
    corpora_df["kws"] = [" ".join(i) for i in combined_kws]
    docs = corpora_df['title'] + corpora_df['abstract'] + corpora_df["kws"]

    if model_type is 'NMF':
        return NMF_cluster(docs, num_topics, num_features, verbose=verbose)
    elif model_type is 'LDA':
        return LDA_cluster(docs, num_topics, num_features, verbose=verbose)
    else:
        raise ValueError("Unknown or unimplemented topic modelling approach!")
    # End if

    raise ValueError("Unknown error occurred!")
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

    if verbose:
        display_topics(nmf, names, num_top_words=10)

    return nmf, trans, names
# End NMF_cluster()


def LDA_cluster(docs, num_topics, num_features, stop_words='english', verbose=True):
    trans, names = cluster_topics(
        CountVectorizer, docs, num_features, stop_words, verbose)

    # Run LDA
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=50,
                                    learning_method='online', learning_offset=50., random_state=0).fit(trans)
    if verbose:
        display_topics(lda, names, num_top_words=10)

    return lda, trans, names
# End LDA_cluster()


def get_topic_by_id(topic_model, trans, topic_id, corpora_df):
    """Get documents related to a topic id.

    Parameters
    ==========
    * topic_id : int, Topic ID (starting from 1)

    Returns
    ==========
    * Pandas DataFrame
    """
    doc_topic = topic_model.transform(trans)

    doc_row_id = []
    for n in range(doc_topic.shape[0]):
        topic_most_pr = doc_topic[n].argmax()

        if topic_most_pr == (topic_id - 1):
            doc_row_id.append(n)
    # End for

    topic_docs = corpora_df.iloc[doc_row_id, :]

    return topic_docs
# End get_topic_by_id()


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
