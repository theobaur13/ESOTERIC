from models import Evidence, EvidenceWrapper
from entity_linking import triple_extraction, levenstein_distance, query_method
from claim_doc_similarity import TF_IDF, cosine_similarity
import os
import sqlite3

class EvidenceRetriever:
    def __init__(self, data_path):
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'wiki-pages.db'))

    def retrieve_evidence(self, claim):
        e_l_docs = self.entity_linking(claim)
        print(e_l_docs)
        evidence_wrapper = EvidenceWrapper()
        evidence_wrapper.add_evidence(Evidence(claim.text, 'This is a test evidence', 0.9, 'test-id'))
        return evidence_wrapper

    def entity_linking(self, claim):
        claim_TF_IDF_df = TF_IDF(claim.text)
        docs = cosine_similarity(claim_TF_IDF_df, self.connection)
        return docs

    def claim_doc_similarity(self, claim):
        # return all doc_ids
        pass