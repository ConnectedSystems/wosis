from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn


def penn_to_wn(tag):
    """
    Convert between a Penn Treebank tag to a simplified Wordnet tag.
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
    """
    Compute the sentence similarity using Wordnet.
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


def jaccard_similarity(sentence1, sentence2):
    """
    Compute the similarity of lemmatized sentences using jaccard similarity scores.
    Example taken from: https://bommaritollc.com/2014/06/30/advanced-approximate-sentence-matching-python/

    :returns: float, score out of 100.0
    """
    sen1 = sentence1.split()
    sen2 = sentence2.split()

    score = 0.0
    denom = float(len(set(sen1).union(sen2)))
    if denom > 0:
        score = len(set(sen1).intersection(sen2)) / denom

    return score * 100
