"""Class that holds topic results"""
import wosis

# wosis.extract_recs

class TopicResult(object):

    def __init__(self, model, trans, feature_names, corpora_df):
        self.model = model
        self._trans = trans
        self.names = feature_names
        self.corpora_df = corpora_df
        self.num_topics = len(self.model.components_)
    # End __init__()

    def get_topic_words(self, num_top_words=10):
        res = {}
        names = self.names
        for topic_idx, topic in enumerate(self.model.components_):
            match = " ".join([names[i]
                            for i in topic.argsort()[:-num_top_words - 1:-1]])
            res[topic_idx+1] = match

        return res
    # End get_topic_words()

    def display_topics(self, num_top_words=10):
        model = self.model
        feature_names = self.names
        matches = self.get_topic_words(num_top_words)
        for i in matches:
            print("Topic {}: {}".format(i, matches[i]))
    # End display_topics()

    def get_topic_by_id(self, topic_id):
        """Get documents related to a topic id.

        Parameters
        ==========
        * topic_id : int, Topic ID (starting from 1)

        Returns
        ==========
        * Pandas DataFrame
        """
        topic_model = self.model
        trans = self._trans
        corpora_df = self.corpora_df

        doc_topic = topic_model.transform(trans)

        doc_row_id = []
        for n in range(doc_topic.shape[0]):
            topic_most_pr = doc_topic[n].argmax()

            if topic_most_pr == (topic_id - 1):
                doc_row_id.append(n)
        # End for

        topic_docs = corpora_df.iloc[doc_row_id, :]

        return topic_docs
    # End get_topic_by_id()

    def find_paper_by_id(self, wos_id):
        """Search for a given record based on its WoS ID

        Parameters
        ==========
        * wos_id : str, Web of Science ID to search for

        Returns
        ==========
        * DataFrame of the matching topic, or None if not found.
        """
        tmp_df = self.corpora_df
        tmp_df.loc[tmp_df.id == wos_id]

        for i in range(self.num_topics):
            topic_id = i + 1
            tmp_topic = self.get_topic_by_id(topic_id)
            match = tmp_topic.loc[tmp_topic.id == wos_id]
            if len(match) > 0:
                print("Found in topic", topic_id)
                return tmp_topic
            # End if
        # End for

    # End find_paper_by_id()
