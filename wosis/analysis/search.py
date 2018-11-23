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


def preview_matches_by_keyword(match_records, specific_keyword=None):
    """
    Parameters
    ==========
    * match_records : dict, records sorted by matching keywords.
    * keywords : list, of keywords

    See Also
    ==========
    * keyword_matches()
    """
    if specific_keyword:
        tmp = {specific_keyword: match_records[specific_keyword]}
    else:
        tmp = match_records
    # End if

    for kw_name in match_records:
        if len(match_records[kw_name]) > 0:
            print('Keyword:' , kw_name)
            for rec in match_records[kw_name]:
                print('  Title:', rec.get('TI'))
                print('  Authors:', '; '.join(rec.get('AU')))
                print('  Journal:', rec.get('SO').title())
                print('  Year:', rec.get('PY'))
                print('  -----------------------------------------------')
            print('===================================================')
        # End if
    # End for

# End preview_matches_by_keyword()


def get_unique_kw_titles(match_records):
    """Get unique titles from a record list.

    Parameters
    ==========
    * match_records : dict, records sorted by matching keywords.
    * keywords : list, of keywords

    Returns
    ==========
    set of unique elements of manuscript titles

    See Also
    ==========
    * keyword_matches()
    """
    titles = set()
    for kw in match_records:
        for rec in match_records[kw]:
            titles.update([rec.get('TI')])
        # End for
    # End for

    return titles
# End get_unique_kw_titles()


def find_pubs_by_authors(records, author_list, threshold=60.0):
    """Get publications by specific authors.

    Parameters
    ==========
    * records : dict, records sorted by matching keywords.
    * author_list : list, of authors
    * threshold : float, similarity of author names have to be above this threshold to be included
                  0 to 100, where 100 is exact match.

    Returns
    ==========
    Metaknowledge Record, set of unique elements of manuscript titles

    See Also
    ==========
    * keyword_matches()
    """
    matching_pubs = {au_i: mk.RecordCollection() for au_i in author_list}
    for rec in records:
        for au, au_i in it.product(rec.authors, author_list):
            # Get first portion of name string
            tmp = au_i.split(' ')[0].split(',')[0].lower()
            inside = tmp in au.lower()
            if inside:
                similar = sims.string_match(au, au_i) > threshold
                if similar:
                    matching_pubs[au_i].add(rec)
                # End if
            # End if
        # End for
    # End for

    for k, rec in matching_pubs.items():
        rec.name = len(rec)

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

        print("{a}\n    {b} ({c})\n    {d}\nKeywords: {e}\n\n{f}\n".format(a=title, b=authors, c=year,
                                                                           d=journal, e=kwds, f=ab))
        print('=========================================')
        count += 1
        if count > num:
            break
    # End for
# End preview_matches()
