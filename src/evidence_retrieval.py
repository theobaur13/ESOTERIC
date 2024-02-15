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
        evidence_wrapper = EvidenceWrapper()

        e_l_docs = self.entity_linking(claim)
        c_d_s_docs = self.claim_doc_similarity(claim)
        for doc in c_d_s_docs:
            id = doc[0]
            score = doc[1]
            evidence = Evidence(claim, None, score, id)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def entity_linking(self, claim):
        pass

    def claim_doc_similarity(self, claim):
        claim_TF_IDF_df = TF_IDF(claim.text)
        docs_scores = cosine_similarity(claim_TF_IDF_df, self.connection)
        return docs_scores