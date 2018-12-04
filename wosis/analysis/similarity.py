from fuzzywuzzy import fuzz


def string_match(word1, word2):
    """Gives a score indicating how well two strings match.

    Parameters
    ==========
    * word1 : str
    * word2 : str

    Returns
    ========
    * float, score between 0 and 100 indicating similarity
             (100 is near-exact match)
    """
    return fuzz.token_sort_ratio(word1, word2)
# End string_match()


def merge_similar(data):
    """Merge n-grams that are positional duplicates.
    e.g. "dam water" and "water dam"

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
