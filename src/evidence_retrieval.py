from models import Evidence, EvidenceWrapper
from entity_linking import triple_extraction, levenstein_distance, query_method
from claim_doc_similarity import TF_IDF, cosine_similarity
import dask.dataframe as dd
import os

class EvidenceRetriever:
    def __init__(self, data_path, batch_size=1000):
        self.data_path = data_path
        self.batch_size = batch_size
        self.last_file_loaded = None
        self.data = self.load_data(data_path, batch_size)
        print(self.data)

    def load_data(self, data_path, batch_size):
        connection = 'sqlite:///' + os.path.join(data_path, 'wiki-pages.db')
        df = dd.read_sql_table('documents', connection, index_col='id', npartitions=1)
        df = df.head(batch_size, compute=False)
        return df

    def retrieve_evidence(self, claim):
        evidence_wrapper = EvidenceWrapper()
        evidence_wrapper.add_evidence(Evidence(claim.text, 'This is a test evidence', 0.9, 'test-id'))
        return evidence_wrapper

    def entity_linking(self, claim):
        pass

    def claim_doc_similarity(self, claim):
        pass