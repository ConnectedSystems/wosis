import os
import wosis.store as store
import wos_parser
import pickle

import time
import warnings

def build_query(inclusive, exclusive, subject_area):
    """Generate a WoS advanced query string

    Parameters
    ==========
    * inclusive
    * exclusive
    * subject_area

    Returns
    ==========
    * str, WoS advanced query string
    """
    query = "("
    query += ' OR '.join('"{}"'.format(w) for w in inclusive)
    query += ")"

    if len(exclusive) > 0:
        query = "(" + query
        query += " NOT ("
        query += " OR ".join('"{}"'.format(w) for w in exclusive)
        query += "))"

    query = "TS=" + query

    if len(subject_area) > 0:
        query += " AND WC=("
        query += " OR ".join('"{}"'.format(subject) for subject in subject_area)
        query += ")"

    return query
# End build_query()


def grab_records(client, query, batch_size=100, verbose=False):
    # Cache results so as to not spam the Clarivate servers
    md5_hash = store.create_query_hash(query)
    cache_file = f'{md5_hash}.dmp'

    os.makedirs('tmp', exist_ok=True)
    cache_file = os.path.join('tmp', cache_file)
    if not os.path.isfile(cache_file):
        recs = []

        probe = client.search(query, count=1)
        q_id = probe.queryId
        num_matches = probe.recordsFound
        print("Found {} records".format(num_matches))

        recs.extend(wos_parser.read_xml_string(probe.records))
        del probe

        # num_matches - `n` as the last loop gets the last `n` records
        # (remember range is end exclusive, hence the -1)
        leap = batch_size - 1
        xml_list = []
        for batch_start in range(2, num_matches, batch_size):
            if verbose:
                print(f'Getting {batch_start} to {batch_start + batch_size - 1}')
            resp = client.retrieve(q_id, batch_size, batch_start)
            recs.extend(wos_parser.read_xml_string(resp.records))
            xml_list.append(resp.records)
        # End for

        with open(cache_file, 'wb') as outfile:
            pickle.dump(xml_list, outfile, pickle.HIGHEST_PROTOCOL)
        # End with
    else:
        with open(cache_file, 'rb') as infile:
            xml_list = pickle.load(infile)
        # End with

        recs = []
        for xml in xml_list:
            recs.extend(wos_parser.read_xml_string(xml))
        # End for
    # End if

    return recs, xml_list
# End grab_records()

def _extract_ref(c_rec, ref_order):
    cr = []
    for detail in ref_order:
        if detail in c_rec.__keylist__:
            cr.append(getattr(c_rec, detail))
    return ", ".join(cr)
# End _extract_ref()


def grab_cited_works(client, query_str, recs, batch_size=100, skip_refs=False, get_all_refs=False):
    os.makedirs('tmp', exist_ok=True)
    md5_hash = store.create_query_hash(query_str)
    cache_file = f'{md5_hash}_ris.dmp'
    cache_file = os.path.join('tmp', cache_file)

    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as infile:
            ris_records = pickle.load(infile)
        # End with

        return ris_records
    # End if

    ris_records = wos_parser.rec_info_to_ris(recs)
    if not skip_refs:
        ris_records = _get_referenced_works(client, ris_records, get_all_refs=get_all_refs)
    else:
        print("Not getting cited records")

    with open(cache_file, 'wb') as outfile:
        pickle.dump(ris_records, outfile, pickle.HIGHEST_PROTOCOL)
    # End with

    return ris_records
# End grab_cited_works()


def _get_referenced_works(client, ris_records, batch_size=100, get_all_refs=False):
    """
    Parameters
    ==========
    * client : WoS Client object
    * ris_records : list, of RIS record objects
    * batch_size : int, number of records to get in one go

    Returns
    ==========
    list, of RIS records with cited references added
    """
    # Get cited articles for each record
    ref_order = ['citedAuthor', 'year', 'citedTitle', 'citedWork', 'volume', 'page', 'docid']

    # NOTE: Where I call sleep is an attempt to avoid overloading the clarivate servers
    for rec in ris_records:
        try:
            cite_recs = client.citedReferences(rec['UT'])
        except:
            time.sleep(2)
            cite_recs = client.citedReferences(rec['UT'])
        # End try

        rec['CR'] = []

        num_refs = cite_recs.recordsFound
        if num_refs == 0:
            continue

        for c_ref in cite_recs.references:
            rec['CR'].append(_extract_ref(c_ref, ref_order))
        # End for

        if (num_refs > batch_size) and get_all_refs:
            q_id = cite_recs.queryId
            warnings.warn("A reference had more than {} citations. This can get quite large...".format(batch_size))

            # num_matches - `n` as the last loop gets the last `n` records
            # (remember range is end exclusive, hence the -1)
            leap = batch_size - 1
            for batch_start in range(batch_size+1, num_refs, batch_size):
                resp = client.citedReferencesRetrieve(q_id, batch_size, batch_start)
                for c_ref in resp:
                    rec['CR'].append(_extract_ref(c_ref, ref_order))
                # End for
            # End for
        else:
            warnings.warn("A reference had more than {0} citations. Only retrieved the first {0}.".format(batch_size))
        # End if
        time.sleep(0.1)
    # End if
    time.sleep(0.1)  # attempting to avoid overloading clarivate servers
    # End for

    print("Finished")
    return ris_records
# End _get_referenced_works()
