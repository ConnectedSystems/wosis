import os
import wosis.store as store
import wos_parser
import wos
import pickle

import time
import warnings

from suds import WebFault


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


def query(queries, overwrite):
    hash_to_query = {}
    for query_str in queries:
        with wos.WosClient(user=wos_config['user'], password=wos_config['password']) as client:
            recs, xml_list = grab_records(client, query_str, verbose=False)
            recs = grab_cited_works(client, query_str, recs, skip_refs=False, get_all_refs=True)
        # End with

        num_ris_records = len(recs)
        print(f"Got {num_ris_records} records")

        md5_hash = store.create_query_hash(query_str)
        hash_to_query.update({md5_hash: query_str})
        prev_q_exists = os.path.isfile(f'{md5_hash}.txt')
        if (prev_q_exists and overwrite) or not prev_q_exists:
            ris_text = wos_parser.to_ris_text(recs)
            wos_parser.write_file(ris_text, f'{md5_hash}', overwrite=overwrite)
        else:
            continue
        # End if
    # End for

    return hash_to_query
# End query()


def grab_records(client, query, batch_size=100, verbose=False):
    """
    Parameters
    ==========
    * client : WoS Client object
    * query : str, Web of Science Advanced Search query string
    * batch_size : int, number of records to request, between 1 and 100 inclusive.
    * verbose : bool, print out more information.

    Returns
    ==========
    * tuple[list], list of parsed XML objects as string, list of XML
    """
    # Cache results so as to not spam the Clarivate servers
    md5_hash = store.create_query_hash(query)
    cache_file = f'{md5_hash}.dmp'

    os.makedirs('tmp', exist_ok=True)
    cache_file = os.path.join('tmp', cache_file)
    if not os.path.isfile(cache_file):
        recs = []

        try:
            probe = client.search(query, count=1)
        except WebFault as e:
            _handle_webfault(client, e)
            probe = client.search(query, count=1)
        # End try

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

            try:
                resp = client.retrieve(q_id, batch_size, batch_start)
            except WebFault as e:
                _handle_webfault(client, e)
                resp = client.retrieve(q_id, batch_size, batch_start)
            # End try

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
    """
    Parameters
    ==========
    * c_rec : WoS Client citation reference object
    * ref_order : list, of reference details in the desired order

    Returns
    ==========
    str, combined reference list in RIS format
    """
    cr = []
    for detail in ref_order:
        if detail in c_rec.__keylist__:
            cr.append(getattr(c_rec, detail))
    return ", ".join(cr)
# End _extract_ref()


def grab_cited_works(client, query_str, recs, batch_size=100, skip_refs=False, get_all_refs=False):
    """
    Parameters
    ==========
    * client : WoS Client
    * query_str : str, WoS advanced query string
    * recs : list, of records
    * batch_size : int, number of records to download each time. Has to be between 1 and 100 inclusive.
    * skip_refs : bool, if False gets references used in each identified publication
    * get_all_refs : bool, if True attempts to get data for all references used in each publication.
                     Otherwise gets the first `batch_size` references.

    Returns
    ==========
    * list, of RIS records
    """
    os.makedirs('tmp', exist_ok=True)
    md5_hash = store.create_query_hash(query_str)
    cache_file = f'{md5_hash}_ris.dmp'
    cache_file = os.path.join('tmp', cache_file)

    if os.path.isfile(cache_file):
        warnings.warn("Using cached results...")
        with open(cache_file, 'rb') as infile:
            ris_records = pickle.load(infile)
        # End with

        return ris_records
    # End if

    ris_records = wos_parser.rec_info_to_ris(recs)
    if not skip_refs:
        warnings.warn("Getting referenced works...")
        ris_records = _get_referenced_works(client, ris_records, get_all_refs=get_all_refs)
    else:
        warnings.warn("Not getting referenced works")

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

    # NOTE: On WebFault exception, we force the program to pause for some time
    # in an attempt to avoid overloading the clarivate servers
    for rec in ris_records:
        try:
            cite_recs = client.citedReferences(rec['UT'])
        except WebFault as e:
            _handle_webfault(client, e)
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

            for batch_start in range(batch_size+1, num_refs, batch_size):
                try:
                    resp = client.citedReferencesRetrieve(q_id, batch_size, batch_start)
                except WebFault as e:
                    _handle_webfault(client, e)
                    resp = client.citedReferencesRetrieve(q_id, batch_size, batch_start)
                # End try

                for c_ref in resp:
                    rec['CR'].append(_extract_ref(c_ref, ref_order))
                # End for
            # End for
        elif (num_refs > batch_size):
            warnings.warn("A reference had more than {0} citations. Only retrieved the first {0}.".format(batch_size))
        # End if
    # End for

    print("Finished")
    return ris_records
# End _get_referenced_works()


def _handle_webfault(client, ex, min_period=3):
    """
    Parameters
    ==========
    client : WoS Client
    ex : WebFault Exception object
    """
    msg = str(ex)

    if "Server.IDLimit" in msg:
        # request threshold exceeded, so reconnect
        print("Server Error Msg:", msg)
        client.close()
        _wait_for_server(2)
        client.connect()
        return
    # End if

    period_msg = "period length is "
    submsg_len = len(period_msg)
    pos = msg.find(period_msg)
    if pos == -1:
        raise RuntimeError("Could not handle WebFault. Error message: {}".format(msg))
    sub_start = pos+submsg_len
    secs_to_wait = max(min_period, int(msg[sub_start:sub_start+3].split(" ")[0]))
    _wait_for_server(secs_to_wait)
# End _handle_webfault()


def _wait_for_server(wait_time=150, verbose=False):
    """
    Parameters
    ==========
    * wait_time : int, number of seconds to wait before attempting HTTP request
    * verbose : bool, print message (default: False)
    """
    if verbose:
        print(f"Waiting {wait_time} second(s) for server to be available again...")
    time.sleep(wait_time)
# End _wait_for_server()
