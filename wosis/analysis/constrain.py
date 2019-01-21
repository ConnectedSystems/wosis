import pandas as pd

import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from rake_nltk import Metric, Rake
from fuzzywuzzy import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from wosis import rc_to_df
from wosis.TopicResult import TopicResult
from wosis.PhraseResult import PhraseResult

import warnings
from zipfile import BadZipfile

__all__ = ['find_topics', 'remove_by_journals', 'remove_by_title', 'remove_empty_DOIs']

# We lemmatize and stem words to homogenize similar content as much as possible
lemmer = WordNetLemmatizer().lemmatize
stemmer = SnowballStemmer('english').stem

def _homogenize(word):
    return stemmer(lemmer(word))


def _ensure_df(corpora):
    if 'metaknowledge' in str(type(corpora)).lower():
        try:
            corpora_df = pd.DataFrame(corpora.forNLP(extraColumns=["AU", "SO", "DE"],
                                                              stemmer=_homogenize))
        except BadZipfile:
            warnings.warn("Could not stem/lemmatize content - set up NLTK WordNet data first!")
            corpora_df = pd.DataFrame(corpora.forNLP(extraColumns=["AU", "SO", "DE"]))
    elif 'dataframe' in str(type(corpora)).lower():
        corpora_df = corpora

    return corpora_df


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
    * tuple, TopicResults object
    """
    corpora_df = _ensure_df(corpora)

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
    corpora_df = _ensure_df(corpora)

    for unrelated in unrelated_journals:
        count_removed = corpora_df.loc[corpora_df['SO'].str.contains(
            unrelated), 'id'].count()
        if verbose:
            print("{}: {}".format(unrelated, count_removed))
        corpora_df = corpora_df.drop(corpora_df.loc[corpora_df['SO'].str.contains(unrelated)].index,
                               axis=0)
    return corpora_df
# End remove_by_journals()


def remove_by_title(corpora, unrelated_titles, verbose=True):
    corpora_df = _ensure_df(corpora)

    for unrelated in unrelated_titles:
        count_removed = corpora_df.loc[corpora_df['title'].str.contains(
            unrelated), 'id'].count()

        if verbose:
            print("{}: {}".format(unrelated, count_removed))

        corpora_df = corpora_df.drop(
            corpora_df.loc[corpora_df['title'].str.contains(unrelated)].index, axis=0)

    return corpora_df
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
    corpora_df = _ensure_df(corpora)
    to_be_removed = corpora_df.loc[corpora['DOI'] == '', :]
    if verbose:
        count_empty = to_be_removed['DOI'].count()
        print("Removing {} records with no DOIs".format(count_empty))

    filtered = corpora_df.loc[corpora_df['DOI'] != '', :]

    return filtered, to_be_removed
# End remove_empty_DOIs()


def find_rake_phrases(corpora, min_len=2, max_len=None, lang='english'):
    """Find interesting phrases in given corpora.
    """
    if 'metaknowledge' in str(type(corpora)).lower():
        corpora_df = pd.DataFrame(corpora.forNLP(extraColumns=["AU", "SO", "DE"], 
                                                 removeCopyright=True))
    else:
        corpora_df = corpora.copy()

    if hasattr(find_phrases, 'rake'):
        rake = find_phrases.rake
    else:
        from nltk.corpus import stopwords
        # rake = Rake(stopwords=stopwords.words(lang), language=lang)
        rake = Rake(min_length=min_len, max_length=max_len, language=lang)
        find_phrases.rake = rake
    # End if

    phrases = corpora_df['abstract'].apply(rake.extract_keywords_from_text, axis=1)

    rake.extract_keywords_from_text()


def find_phrases(corpora, top_n=5, verbose=False):
    """Find interesting phrases.

    Inspired by work conducted by Rabby et al. (2018)

    * A Flexible Keyphrase Extraction Technique for Academic Literature

    This approach attempts to identify phrases of interest by identifying sentences
    with similar, repeating, elements throughout the text.
    
    Conceptually, 
    * the name of a method/approach may be introduced, discussed, and mentioned again in the conclusion.
    * Important findings may be framed, findings alluded to, and then discussed.


    Parameters
    ==========
    * corpora : Pandas DataFrame
    * top_n : int, number of phrases to display
    * verbose : bool, if True prints text, document title and top `n` phrases. Defaults to False.

    Returns
    ==========
    * dict, results with DOI has main key, human readable document title and DOI as sub-keys and identified phrases as elements
    """
    ccc = corpora['abstract'].tolist()
    results = {}
    for c_idx, corpus in enumerate(ccc):
        sent_tokenize_list = nltk.sent_tokenize(corpus)
        sent_corpora = [[sent, 0.0] for sent in sent_tokenize_list]

        sent_score = pd.DataFrame({'text': sent_tokenize_list})
        sent_score['score'] = 0.0

        doc_title = corpora.iloc[c_idx]['title'] + " ({})".format(corpora.iloc[c_idx]['year'])
        if verbose:
            print(doc_title)

        if len(corpus) == 0:
            print("    No abstract for {}, skipping...\n".format(doc_title))
            return None

        for idx, sent in enumerate(sent_tokenize_list):
            split_sent = sent.split(" ")

            sent_len = len(split_sent)
            if sent_len % 2 == 0:
                central_pos = int((sent_len - 1) / 2)
            else:
                central_pos = int(sent_len / 2)

            root = split_sent[central_pos]

            for candidate_sentence in sent_corpora:
                candidate = candidate_sentence[0]
                if (sent == candidate) or (root not in candidate):
                    continue

                current_score = sent_score.loc[idx, 'score']
                sent_score.loc[idx, 'score'] = current_score + (fuzz.token_set_ratio(sent, candidate) / 100.0)
            # End for
        # End for

        sent_score = sent_score[sent_score['score'] > 0.0]
        sent_score = sent_score.sort_values('score', ascending=False).reset_index(drop=True)
        tmp = sent_score[0:top_n]

        results[corpora.iloc[c_idx]['DOI']] = {
            'doc_title': doc_title,
            'phrases': tmp.to_dict(),
            'wos_id': corpora.iloc[c_idx]['id']
        }

        if verbose:
            for row in tmp.itertuples():
                print(row.text, '\n(Score: {})'.format(row.score), '\n')

        if verbose:
            print('='*20, '\n')

    return PhraseResult(results)


# def scan_for_phrases(corpora):
#     corpora_df = wosis.rc_to_df(corpora)
#     all_abstracts = corpora_df['abstract'].tolist()
#     results = {}
#     for row in corpora_df.itertuples():
#         c_idx = row.Index
#         corpus = row.abstract
#         identify_phrases(corpus)
#         results[row.title]



def rabby_find_phrase(corpus):
    """Implementation based on paper by Rabby et al. (2018)

    * A Flexible Keyphrase Extraction Technique for Academic Literature
    """
    raise NotImplementedError("Unfinished, do not use")

    candidate_patterns = """
    NP:
        {(<NN.*>+)|(<NN.*>+<JJ.*>?)|(<JJ.*>?<NN.*>+)}
        {<NN.*|JJ>*<NN.*>}
        {(<JJ.>|<NN.>)*<IN>?(<JJ.>|<NN.>)*<NN.>}
        {<PRP>?<JJ.*>*<NN.*>+}
        {<DT|PP$>?<JJ>*<NN.*>+}
        {(<\w+DT>)?(<\w+JJ>)*(<\w+>(<NN|NP|PRN>))}
        {(<NN.*>+<JJ.*>?)|(<JJ.*>?<NN.*>+)}
    """

    cp = nltk.RegexpParser(grammar)

    sent_tokenize_list = nltk.sent_tokenize(corpus)

    for sent in sent_tokenize_list:
        tagged_words = nltk.word_tokenize(sent)
        tagged_tokens = nltk.tag.pos_tag(tagged_words)
        identified = cp.parse(tagged_tokens)
        
        sent_len = len(sent.split(" "))
        if sent_len % 2 == 0:
            central_pos = int((sent_len - 1) / 2)
        else:
            central_pos = int(sent_len / 2)
