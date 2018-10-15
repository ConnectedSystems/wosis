import metaknowledge as mk
from . import similarity as sims
import itertools as it

def search_records(records, keywords, threshold=60.0):
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
                    if sims.string_match(kwi, kw) > threshold:
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

    matches.name = str(len(matches))

    return matches
# End search_records()


def keyword_matches(records, keywords, threshold=60.0):
    """
    Get records for each individiual keyword of interest

    Parameters
    ==========
    * records : iterable, of RIS records
    * keywords : list[str], of keywords

    Returns
    ==========
    * tuple[dict], matching records by keyword, and {keyword: number of matching records}
    """
    matching_records = {}
    summary = {}
    for kw in keywords:
        matching_records[kw] = search_records(records, set([kw, ]), threshold)
        summary[kw] = len(matching_records[kw])
    # End for

    return matching_records, summary
# End keyword_matches()


def find_pubs_by_authors(records, author_list, threshold=60.0):
    matching_pubs = {au_i: mk.RecordCollection() for au_i in author_list}
    for rec in records:
        for au, au_i in it.product(rec.authors, author_list):
            # if au_i.split(' ')[1] in au:
            #     print(au, "|", au_i, ":", sims.string_match(au, au_i))

            if sims.string_match(au, au_i) > threshold:
                matching_pubs[au_i].add(rec)
            # End if
        # End for
        # break
    # End for

    return matching_pubs
# End find_pubs_by_authors()


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

        tmp = rec.get('DE') + rec.get('ID')
        kwds = '; '.join([kw.strip() for kw in tmp if kw.strip() != ''])
        journal = rec.get('SO')

        ab = rec.get('AB', None)

        if limit_abstract:
            ab = ab[0:limit_abstract]

        print(f"{title}\n    {authors} ({year})\n    {journal}\nKeywords: {kwds}\n\n{ab}\n")
        print('=========================================')
        count += 1
        if count > num:
            break
    # End for
# End preview_matches()
