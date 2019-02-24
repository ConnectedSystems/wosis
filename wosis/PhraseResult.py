import pandas as pd

class PhraseResult(object):
    """Holds Phrase Extraction results"""

    def __init__(self, phrase_results):
        """
        Parameters
        ==========
        * recs : dict, {WoS : MetaKnowledge RecordCollection}
        """
        self.phrases = phrase_results

    # End __init__()

    @property
    def documents(self):
        return {k: self.phrases[k]['doc_title']
                for k in self.phrases.keys()}

    @property
    def document_ids(self):
        return [self.phrases[i]['wos_id'] for i in self.phrases]

    @property
    def titles(self):
        titles = []
        for k, entry in self.phrases.items():
            titles.append(entry['doc_title'])
        return titles

    def get_phrases(self, doi):
        """Get key phrases for a specific DOI"""
        selected = self.phrases[doi]
        print(selected['doc_title'])
        return pd.DataFrame(selected['phrases'])

    @property
    def all_phrases(self):
        tmp = None
        num_phrases = 0
        for doi in self.phrases:
            current = self.phrases[doi]
            doctitle = current['doc_title']
            phrases = current['phrases']['text']
            extracted = pd.DataFrame.from_dict({(doi, doctitle): phrases},
                            orient='index')
            tmp = pd.concat((tmp, extracted), axis=0)
            num_phrases = max(len(phrases), num_phrases)
        # End for

        tmp.columns = [i for i in range(1, num_phrases+1)]
        tmp.index.names = ('DOI', 'title')

        return tmp

    def display_phrases(self):
        for doi, doc in self.phrases.items():
            print(doc['doc_title'], '\nhttps://dx.doi.org/{}'.format(doi), '\n')
            for phrases in doc['phrases']['text'].values():
                print(phrases, '\n')
            print("="*20, '\n')

    def __len__(self):
        return len(self.phrases)
