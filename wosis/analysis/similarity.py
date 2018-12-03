from nltk import pos_tag, word_tokenize, edit_distance
from nltk.corpus import wordnet as wn

from fuzzywuzzy import fuzz


def penn_to_wn(tag):
    """Convert between a Penn Treebank tag to a simplified Wordnet tag.
    Examples taken from: http://nlpforhackers.io/wordnet-sentence-similarity/
    """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    """Examples taken from: http://nlpforhackers.io/wordnet-sentence-similarity/"""
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except IndexError:
        return None


def sentence_similarity(sentence1, sentence2):
    """Compute the sentence similarity using Wordnet.
    Examples taken from: http://nlpforhackers.io/wordnet-sentence-similarity/
    """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) if synset.path_similarity(ss)
                          is not None else 0.0 for ss in synsets2])

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    if count > 0:
        score /= count

    return score
# End sentence_similarity()


def string_match(word1, word2):
    """Gives a score indicating how well two strings match.

    Parameters
    ==========
    * word1 : str
    * word2 : str

    Returns
    ========
    * float, score between 0 and 100 indicating similarity (100 is near-exact match)
    """
    return fuzz.token_sort_ratio(word1, word2)
# End string_match()


def merge_similar(data):
    """Merge n-grams that are positional duplicates, e.g. "dam water" and "water dam"

    Parameters
    ==========
    * data : list, of keywords/ngrams

    Returns
    ========
    * list, of merged keywords/ngrams
    """
    dataset = [set(k) for k in data]
    tmp = [k for k in dataset if dataset.count(k) == 1]

    tmp2 = []
    for i in dataset:
        if i not in tmp and i not in tmp2:
            tmp2.append(i)
        # End if
    # End for

    unique_items = tmp + tmp2

    final = {}
    for it in unique_items:
        final[tuple(it)] = 0
        for entry, v in data.items():
            if set(entry) == it:
                final[tuple(it)] += v
            # End if
        # End for
    # End for

    return final
# End merge_similar()
