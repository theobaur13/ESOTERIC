import os
import sqlite3
from models import Evidence, EvidenceWrapper
from legacy.FEVERISH_1.utils.claim_doc_similarity import TF_IDF, cosine_similarity
from legacy.FEVERISH_1.utils.document_retrieval import FAISS_search

class EvidenceRetriever:
    def __init__(self, data_path):
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'data.db'))

    def retrieve_evidence(self, claim):
        evidence = self.retrieve_documents(claim)
        evidence = self.retrieve_passages(evidence)
        return evidence
    
    def retrieve_documents(self, claim):
        evidence_wrapper = EvidenceWrapper(claim)
        # c_d_dict = FAISS_search(claim, self.data_path)

        cursor = self.connection.cursor()
        # for doc_id, score in c_d_dict.items():
        #     cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
        #     text = cursor.fetchone()[0]
        #     evidence = Evidence(claim, text, score, doc_id, doc_retrieval_method="FAISS")
        #     evidence_wrapper.add_evidence(evidence)

        for doc_id, score in self.lexical_similarity(claim):
            cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
            text = cursor.fetchone()[0]
            if not evidence_wrapper.get_evidence_by_id(doc_id):
                evidence = Evidence(claim, text, score, doc_id, doc_retrieval_method="TF-IDF")
                evidence_wrapper.add_evidence(evidence)
            else:
                evidence_wrapper.remove_evidence(doc_id)
                evidence = Evidence(claim, text, score, doc_id, doc_retrieval_method="FAISS & TF-IDF")
                evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper
    
    def retrieve_passages(self, evidence_wrapper):
        for evidence in evidence_wrapper.get_evidences():
            evidence.set_evidence_sentence("Placeholder for sentence retrieval.")
        return evidence_wrapper

    def lexical_similarity(self, claim):
        claim_TF_IDF_df = TF_IDF(claim)
        docs = cosine_similarity(claim_TF_IDF_df, self.connection)
        return docs