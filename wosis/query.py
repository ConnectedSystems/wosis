import os
from os.path import join as pj
from glob import glob
from io import StringIO

import wosis
import wosis.store as store
import wos_parser
import wos
import pickle
import re

import metaknowledge as mk
import pandas as pd

import time
import warnings
import yaml
import json
from tqdm.auto import tqdm

from suds import WebFault

import logging


__all__ = ['load_config', 'build_query', 'query', 'grab_records',
           'grab_cited_works', 'get_citing_works', 'load_query_results', 'get_num_citations']

start_rec = re.compile("<REC", re.IGNORECASE)
end_rec = re.compile("</REC>", re.IGNORECASE)

# suppress output from suds which gets really annoying
logging.getLogger('suds.client').setLevel(logging.CRITICAL)


def load_config(config_file):
    """Load a YAML-based configuration file.

    Parameters
    ==========
    * config_file : str, path to configuration file.

    Returns
    ==========
    * wos_config : dict, of 'user' and 'password' key-values
    """
    with open(config_file) as config:
        wos_config = yaml.load(config)
        wos_config = wos_config['wos']

    return wos_config
# End load_config()


def build_query(search_params):
    """Generate a WoS advanced query string

    Parameters
    ==========
    * search_params : dict
        * inclusive_kw : inclusive keywords
        * exclusive_kw : exclusive keywords
        * inclusive_jo : search within specific journals
        * exclusive_jo : exclude specific journals
        * subject_area : subject area

    Returns
    ==========
    * str, WoS advanced query string
    """
    query = ''
    if 'inclusive_kw' in search_params:
        query = "("
        query += ' OR '.join('"{}"'.format(w)
                             for w in search_params['inclusive_kw'])
        query += ")"

    if 'exclusive_kw' in search_params:
        query = "(" + query
        query += " NOT ("
        query += " OR ".join('"{}"'.format(w)
                             for w in search_params['exclusive_kw'])
        query += "))"

    query = "TS=" + query

    if 'inclusive_jo' in search_params:
        query += " AND SO=("
        query += " OR ".join('"{}"'.format(jo)
                             for jo in search_params['inclusive_jo'])
        query += ")"

    if 'exclusive_jo' in search_params:
        if 'inclusive_jo' in search_params:
            query += " NOT ("
        else:
            query += " NOT SO=("
        query += " OR ".join('"{}"'.format(jo)
                             for jo in search_params['exclusive_jo'])
        query += ")"

    if 'subject_area' in search_params:
        query += " AND WC=("
        query += " OR ".join('"{}"'.format(subject)
                             for subject in search_params['subject_area'])
        query += ")"

    return query
# End build_query()


def load_query_results(fn, file_loc=None):
    """Load a stored query result saved in RIS format.

    Parameters
    ==========
    * fn : str, path to file, including path and extension.
    """
    if file_loc:
        raise ValueError("file location parameter deprecated - specify path and extension in filename instead")

    if not os.path.isfile(fn):
        raise ValueError("Could not find file - did you forget the folder or file extension?")

    return mk.RecordCollection(fn)
# End load_query_results()



def query(queries, overwrite, config, time_span=None, tmp_dir='tmp', skip_refs=False):
    """Query the Web of Science collection via its API.

    Parameters
    ==========
    * queries : list[str], list of Web of Science Advanced Search query string
    * overwrite : bool, overwrite previous identical search if it exists.
    * config : dict, Web of Science configuration
    * time_span : None or Dict,
                  begin - Beginning date for this search. Format: YYYY-MM-DD
                  end - Ending date for this search. Format: YYYY-MM-DD
    * tmp_dir : str, path to temporary directory. Defaults to `tmp` in current location.
    * skip_refs : bool, get bibliometric data for works referenced in the corpora as well.
                  Defaults to False.

    Returns
    ==========
    * tuple[dict]:
        * hash_to_query: query_id to query string
        * hash_to_col: query_id to metaknowledge collection
    """
    hash_to_query = {}
    hash_to_col = {}
    os.makedirs(tmp_dir, exist_ok=True)  # Create temp dir if necessary
    for query_str in queries:
        cache_fn = store.create_query_hash(query_str, time_span)
        hash_to_query.update({cache_fn: query_str})
        tmp_file = pj(tmp_dir, cache_fn)

        ris_file = '{}_ris.txt'.format(tmp_file)
        prev_q_exists = os.path.isfile(ris_file)
        if (prev_q_exists and overwrite) or not prev_q_exists:

            with wos.WosClient(user=config['user'], password=config['password']) as client:
                recs, xml_list = grab_records(client, query_str, time_span=time_span, 
                                              verbose=False)
                if not skip_refs:
                    recs = grab_cited_works(
                        client, query_str, recs, skip_refs=False, get_all_refs=True)
            # End with

            num_ris_records = len(recs)
            print("Got {} records".format(num_ris_records))

            # store.write_cache(recs, ris_file)

            if not isinstance(recs[0], dict):
                ris_info = wos_parser.rec_info_to_ris(recs)
            else:
                ris_info = recs
            # End if

            ris_text = wos_parser.to_ris_text(ris_info)
            wos_parser.write_file(
                ris_text, ris_file, overwrite=overwrite, ext='')
        # End if

        RC = mk.RecordCollection(ris_file)
        hash_to_col[cache_fn] = RC
    # End for

    # dump out the query hash to file
    with open(pj(tmp_dir, 'hash_to_query.txt'), 'w') as hash_file:
        json.dump(hash_to_query, hash_file, indent=2)

    return hash_to_query, hash_to_col
# End query()


def _ensure_separation(xml_string):
    xml_string = start_rec.sub("\n<REC", xml_string)
    xml_string = end_rec.sub("</REC>\n", xml_string)
    return xml_string

def grab_records(client, query, batch_size=100, time_span=None,
                 verbose=False):
    """Retrieves publication records in raw XML format. Use via `query()`.

    Parameters
    ==========
    * client : WoS Client object
    * query : str, Web of Science Advanced Search query string
    * batch_size : int, number of records to request, between 1 and 100 inclusive.
    * time_span : None or Dict,
                  begin - Beginning date for this search. Format: YYYY-MM-DD
                  end - Ending date for this search. Format: YYYY-MM-DD
    * verbose : bool, print out more information.

    Returns
    ==========
    * tuple[list], list of parsed XML objects as string, list of XML
    """
    recs = []
    try:
        probe = client.search(query, count=1, timeSpan=time_span)
    except WebFault as e:
        _handle_webfault(client, e)
        probe = client.search(query, count=1, timeSpan=time_span)
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
    for batch_start in tqdm(range(2, num_matches, batch_size)):
        if verbose:
            print('Getting {} to {}'.format(
                batch_start, batch_start + leap))

        try:
            resp = client.retrieve(q_id, batch_size, batch_start)
        except WebFault as e:
            _handle_webfault(client, e)
            resp = client.retrieve(q_id, batch_size, batch_start)
        # End try

        xml = _ensure_separation(resp.records)
        recs.extend(wos_parser.read_xml_string(xml))
        xml_list.append(xml)
    # End for

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


def grab_cited_works(client, query_str, recs, time_span=None, batch_size=100, skip_refs=False,
                     get_all_refs=False):
    """Retrieves citations within publications in raw XML format. Used in `query()`.

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
    ris_records = wos_parser.rec_info_to_ris(recs)
    if not skip_refs:
        warnings.warn("Getting referenced works for {} publications...".format(len(recs)))

        ris_records = _get_referenced_works(
            client, ris_records, get_all_refs=get_all_refs)
    else:
        warnings.warn("Not getting referenced works")

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
    ref_order = ['citedAuthor', 'year', 'citedTitle',
                 'citedWork', 'volume', 'page', 'docid']

    # NOTE: On WebFault exception, we force the program to pause for some time
    # in an attempt to avoid overloading the clarivate servers
    for rec in tqdm(ris_records):
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
            warnings.warn(
                "A reference had more than {} citations. This can take a long time and the cache file can get quite large...".format(batch_size))

            for batch_start in range(batch_size + 1, num_refs, batch_size):
                try:
                    resp = client.citedReferencesRetrieve(
                        q_id, batch_size, batch_start)
                except WebFault as e:
                    _handle_webfault(client, e)
                    resp = client.citedReferencesRetrieve(
                        q_id, batch_size, batch_start)
                except Exception as e:
                    # Handle URLError
                    _handle_webfault(client, e, min_period=60)
                    resp = client.citedReferencesRetrieve(
                        q_id, batch_size, batch_start)
                # End try

                for c_ref in resp:
                    rec['CR'].append(_extract_ref(c_ref, ref_order))
                # End for
            # End for
        elif (num_refs > batch_size):
            warnings.warn(
                "A reference had more than {0} citations. Only retrieved the first {0}.".format(batch_size))
        # End if
    # End for

    # print("Finished")
    return ris_records
# End _get_referenced_works()


def get_citing_works(wos_id, config, batch_size=100, cache_as=None, overwrite=True):
    """Retrieve publications that cite a given paper

    Parameters
    ==========
    * wos_id : str, Web of Science ID of publication to get data for
    * config : dict, config settings
    * batch_size : int, number of records to retrieve in a single request
    * cache_as : str, location of cache file to use
    * overwrite : bool, overwrite cache file or not

    Returns
    ==========
    * Metaknowledge RecordCollection
    """
    if cache_as:
        file_list = glob(cache_as)
        if file_list:
            return mk.RecordCollection("{}.txt".format(cache_as))

    with wos.WosClient(user=config['user'], password=config['password']) as client:

        try:
            # Using a count of 0 returns just the summary information
            probe = client.citingArticles(wos_id, count=0)
        except Exception as e:
            _handle_webfault(client, e)
            probe = client.citingArticles(wos_id, count=0)
        # End try

        recs = []
        num_recs = probe.recordsFound

        print("Found {} records".format(num_recs))

        for batch in tqdm(range(1, num_recs, batch_size)):
            try:
                # Using a count of 0 returns just the summary information
                probe = client.citingArticles(wos_id, count=batch_size, offset=batch)
            except Exception as e:
                _handle_webfault(client, e)
                probe = client.citingArticles(wos_id, count=batch_size, offset=batch)
            # End try

            xml = _ensure_separation(probe.records)
            recs.extend(wos_parser.read_xml_string(xml))

    ris_info = wos_parser.rec_info_to_ris(recs)
    ris_text = wos_parser.to_ris_text(ris_info)
    wos_parser.write_file(
        ris_text, cache_as, overwrite=overwrite)

    RC = mk.RecordCollection("{}.txt".format(cache_as))

    return RC
# End get_citing_works()


def get_num_citations(records, config, cache_as=None):
    """Send query to get the number of citations for a given WoS record.

    Parameters
    ==========
    * records : Metaknowledge RecordCollection
    * config : dict, config settings
    * cache_as : str or None, if specified use data to specified file (or save to it for later reuse)

    Returns
    ==========
    * Pandas DataFrame, publication details with citations
    """
    if cache_as:
        file_list = glob(cache_as)
        if file_list:
            return pd.read_csv(cache_as, index_col=0)

    cites = {}
    with wos.WosClient(user=config['user'], password=config['password']) as client:
        for rec in tqdm(records):
            wos_id = rec.get('id')
            try:
                # Using a count of 0 returns just the summary information
                probe = client.citingArticles(wos_id, count=0)
            except Exception as e:
                _handle_webfault(client, e)
                probe = client.citingArticles(wos_id, count=0)
            # End try

            cites[wos_id] = probe.recordsFound
        # End for

    for wos_id in cites:
        rec = records.getID(wos_id)
        rec.TC = cites[wos_id]

    citations = pd.DataFrame({'citations': list(cites.values()), "id": list(cites.keys())})
    tmp_df = wosis.rc_to_df(records)
    results_df = (pd.merge(tmp_df, citations, on='id')).sort_values('citations', ascending=False).reset_index(drop=True)

    if cache_as:
        results_df.to_csv(cache_as)

    return results_df


def _handle_webfault(client, ex, min_period=3):
    """
    Parameters
    ==========
    client : WoS Client
    ex : WebFault Exception object
    """
    msg = str(ex)

    period_msg = "period length is "
    submsg_len = len(period_msg)
    pos = msg.find(period_msg)
    if pos == -1:
        if "Server.IDLimit" in msg:
            # request threshold exceeded, so reconnect
            print("Server Error Msg:", msg)

            client.close()
            _wait_for_server(3)
            client.connect()
            _wait_for_server(1)
            return
        # End if

        if "Back-end server is at capacity" in msg or "URLError" in msg:
            # have to wait a bit...
            client.close()
            _wait_for_server(60)
            client.connect()
            return
        # End if

    else:
        sub_start = pos + submsg_len
        secs_to_wait = max(min_period, int(
            msg[sub_start:sub_start + 3].split(" ")[0]))
        _wait_for_server(secs_to_wait)

        return
    # End if

    raise RuntimeError(
        "Could not handle WebFault. Error message: {}".format(msg))
# End _handle_webfault()


def _wait_for_server(wait_time=150, verbose=False):
    """
    Parameters
    ==========
    * wait_time : int, number of seconds to wait before attempting HTTP request
    * verbose : bool, print message (default: False)
    """
    if verbose:
        print("Waiting {} second(s) for server to be available again...".format(wait_time))
    time.sleep(wait_time)
# End _wait_for_server()
