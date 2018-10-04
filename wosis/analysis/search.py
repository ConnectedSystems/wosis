import metaknowledge as mk
from . import similarity as sims


def search_records(records, keywords):
    matches = mk.RecordCollection()
    for record in records:
        kwds = record.get('keywords', None)
        abstract = record.get('AB', None)

        if kwds:
            subset = keywords.intersection(set([kw.lower() for kw in kwds]))
            if len(subset) > 0:
                matches.add(record)
            else:
                # attempt to match on jaccard similarity
                tmp = [kw.lower() for kw in kwds]
                combinations = [(a, b) for a in keywords for b in tmp]
                for kwi, kw in combinations:
                    if sims.jaccard_similarity(kwi, kw) > 60:
                        # print("Jaccard match found: {} | {}".format(kwi, kw))
                        if record not in matches:
                            matches.add(record)
                    # End if
                # End for
            # End if
        # End if

        if record not in matches and abstract is not None:
            tmp = abstract.lower()
            for kwi in keywords:
                if tmp.find(kwi) > -1:
                    matches.add(record)
                    break
                # End if
        # End if
    # End for

    return matches
# End search_records()


def preview_matches(search_results, num=5, limit_abstract=None):
    """
    Parameters
    ==========
    * search_results : iterable, of RIS records
    * num : int, number of records to preview
    * limit_abstract : int, Number of characters to display in the abstract. Defaults to None.
    """
    count = 0
    for rec in search_results:
        title = rec.title
        year = rec.get('PY', None)
        authors = rec.get('AU', None)
        if authors:
            authors = "; ".join(authors)
        year = '- No Publication Year -' if not year else year

        kwds = ' '.join(['Keywords:'] + rec.get('DE') + rec.get('ID'))
        journal = rec.get('SO')

        ab = rec.get('AB', None)

        if limit_abstract:
            ab = ab[0:limit_abstract]

        print(f"{title}\n    {authors} ({year})\n    {journal}\n{kwds}\n\n{ab}\n")
        print('=========================================')
        count += 1
        if count > num:
            break
