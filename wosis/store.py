import hashlib

import wos_parser
import json

def create_query_hash(query_str):
    hash_object = hashlib.md5(query_str.encode())
    md5_hash = hash_object.hexdigest()
    return md5_hash
# End create_query_hash()


def export_ris_file(records, filename):
    ris_text = wos_parser.to_ris_text(records)
    wos_parser.write_file(ris_text, filename)
# End export_ris_file()

def store_query_hash(hash_to_query, fn='hash_to_query.txt'):
    # use `json.loads` to do the reverse
    with open(fn, 'w') as file:
         file.write(json.dumps(hash_to_query, indent=2))
    # End with
# End store_query_hash()
