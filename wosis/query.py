import os
from os.path import join as pj
from glob import glob

import wosis
import wosis.store as store
import wos_parser
import wos
import pickle

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


def load_query_results(query_id, file_loc='tmp'):
    fn = pj(file_loc, query_id+'.txt')
    return mk.RecordCollection(fn)
# End load_query_results()



def query(queries, overwrite, config, time_span=None, tmp_dir='tmp'):
    """Query the Web of Science collection via its API.

    Parameters
    ==========
    * queries : WoS Client object
    * overwrite : str, Web of Science Advanced Search query string
    * config : int, number of records to request, between 1 and 100 inclusive.
    * time_span : None or Dict,
                  begin - Beginning date for this search. Format: YYYY-MM-DD
                  end - Ending date for this search. Format: YYYY-MM-DD
    * tmp_dir : str, path to temporary directory. Defaults to `tmp` in current location.

    Returns
    ==========
    * tuple[dict]:
        * hash_to_query: query_id to query string
        * hash_to_col: query_id to metaknowledge collection
    """
    hash_to_query = {}
    hash_to_col = {}
    for query_str in queries:
        with wos.WosClient(user=config['user'], password=config['password']) as client:
            recs, xml_list = grab_records(
                client, query_str, time_span=time_span, verbose=False)
            recs = grab_cited_works(
                client, query_str, recs, skip_refs=False, get_all_refs=True)
        # End with

        num_ris_records = len(recs)
        print("Got {} records".format(num_ris_records))

        md5_hash = store.create_query_hash(query_str)
        if time_span:
            md5_hash = "{}_{}-{}".format(md5_hash, time_span['begin'], time_span['end'])
        hash_to_query.update({md5_hash: query_str})
        tmp_file = pj(tmp_dir, md5_hash)
        prev_q_exists = os.path.isfile('{}.txt'.format(tmp_file))
        if (prev_q_exists and overwrite) or not prev_q_exists:
            ris_text = wos_parser.to_ris_text(recs)
            wos_parser.write_file(
                ris_text, '{}'.format(tmp_file), overwrite=overwrite)
        # else:
        #     continue
        # # End if

        RC = mk.RecordCollection("{}.txt".format(tmp_file))
        hash_to_col[md5_hash] = RC
    # End for

    # dump out the query hash to file
    with open(pj(tmp_dir, 'hash_to_query.txt'), 'w') as hash_file:
        hash_file.write(json.dumps(hash_to_query, indent=2))

    return hash_to_query, hash_to_col
# End query()


def grab_records(client, query, batch_size=100, time_span=None, tmp_dir='tmp',
                 verbose=False):
    """
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
    # Cache results so as to not spam the Clarivate servers
    md5_hash = store.create_query_hash(query)

    os.makedirs(tmp_dir, exist_ok=True)
    cache_file = '{}.dmp'.format(md5_hash)
    cache_file = os.path.join(tmp_dir, cache_file)
    if not os.path.isfile(cache_file):
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


def grab_cited_works(client, query_str, recs, batch_size=100, skip_refs=False,
                     get_all_refs=False, tmp_dir='tmp'):
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
    os.makedirs(tmp_dir, exist_ok=True)
    md5_hash = store.create_query_hash(query_str)
    cache_file = '{}_ris.dmp'.format(md5_hash)
    cache_file = os.path.join(tmp_dir, cache_file)

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

        ris_records = _get_referenced_works(
            client, ris_records, get_all_refs=get_all_refs)
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


def get_citing_works(wos_id, config):
    """Retrieve publications that cite a given paper

    Parameters
    ==========
    * wos_id : str, Web of Science ID
    * config : dict, config settings

    Returns
    ==========
    * Metaknowledge RecordCollection
    """
    raise NotImplementedError("This method is not yet finished")
    with wos.WosClient(user=config['user'], password=config['password']) as client:
        pass
        # client.
    # End with
# End get_citing_works()


def get_num_citations(records, config, cache_dir=None):
    """Send query to get the number of citations for a given WoS record.

    Parameters
    ==========
    * records : Metaknowledge RecordCollection
    * config : dict, config settings
    * cache_dir : str or None, if specified use data in cache directory (or save data to it).
                  Defaults to None.

    Returns
    ==========
    * Pandas DataFrame, publication details with citations
    """
    if cache_dir:
        record_name = records.name
        if "empty" in record_name.lower():
            print("Cannot cache results - specify a name for the Record Collection!")
            fn = None
        else:
            fn = '{}/{}_citations.csv'.format(cache_dir, record_name)
            file_list = glob(fn)
            if file_list:
                return pd.read_csv(fn, index_col=0)


    cites = {}
    with wos.WosClient(user=config['user'], password=config['password']) as client:
        for rec in tqdm(records):
            wos_id = rec.get('id')
            try:
                probe = client.citingArticles(wos_id, count=1)
            except Exception as e:
                _handle_webfault(client, e)
                probe = client.citingArticles(wos_id, count=1)
            # End try

            cites[wos_id] = probe.recordsFound
        # End for

    for wos_id in cites:
        rec = records.getID(wos_id)
        rec.TC = cites[wos_id]

    citations = pd.DataFrame({'citations': list(cites.values()), "id": list(cites.keys())})
    tmp_df = wosis.rc_to_df(records)
    results_df = (pd.merge(tmp_df, citations, on='id')).sort_values('citations', ascending=False).reset_index(drop=True)

    if cache_dir and fn:
        results_df.to_csv(fn)

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
