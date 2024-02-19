import os
import sqlite3
import spacy
from models import Evidence, EvidenceWrapper
from entity_linking import triple_extraction, FAISS_search
from claim_doc_similarity import TF_IDF, cosine_similarity

class EvidenceRetriever:
    def __init__(self, data_path):
        print("Initialising EvidenceRetriever")
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'wiki-pages.db'))
        self.triple_extraction_model = spacy.load('en_core_web_sm')

    def retrieve_evidence(self, claim):
        evidence_wrapper = EvidenceWrapper()

        e_l_doc_ids = self.entity_linking(claim)
        cursor = self.connection.cursor()
        for doc_id in e_l_doc_ids:
            cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
            text = cursor.fetchone()[0]
            evidence = Evidence(claim, text, None, doc_id)
            evidence_wrapper.add_evidence(evidence)

        # c_d_s_docs = self.claim_doc_similarity(claim)
        # for doc in c_d_s_docs:
        #     id = doc[0]
        #     score = doc[1]
        #     evidence = Evidence(claim, None, score, id)
        #     evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def entity_linking(self, claim):
        print("Starting document retrieval by entity linking")
        triples = triple_extraction(claim.text, self.triple_extraction_model)
        doc_ids = FAISS_search(claim.text, self.data_path)
        return doc_ids

    def claim_doc_similarity(self, claim):
        print("Starting document retrieval by claim-document similarity")
        claim_TF_IDF_df = TF_IDF(claim.text)
        docs_scores = cosine_similarity(claim_TF_IDF_df, self.connection)
        return docs_scores