import os
import sqlite3
import spacy
import claucy
from transformers import pipeline
from models import Evidence, EvidenceWrapper
from evidence_retrieval.tools.document_retrieval import triple_extraction, FAISS_search
from evidence_retrieval.tools.passage_retrieval import answerability_filter, passage_extraction

class EvidenceRetriever:
    def __init__(self, data_path):
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'wiki-pages.db'))
        self.triple_extraction_model = spacy.load('en_core_web_sm')
        self.answerability_pipe = pipeline("text-classification", model="potsawee/longformer-large-4096-answerable-squad2")
        claucy.add_to_pipe(self.triple_extraction_model)

    def retrieve_evidence(self, claim):
        evidence = self.retrieve_documents(claim)
        evidence = self.retrieve_passages(evidence)
        return evidence

    def retrieve_documents(self, claim):
        print("Starting document retrieval for claim: '" + str(claim.text) + "'")
        evidence_wrapper = EvidenceWrapper(claim)
        # triples = triple_extraction(claim.text, self.triple_extraction_model)
        e_l_dict = FAISS_search(claim.text, self.data_path)

        cursor = self.connection.cursor()
        for doc_id, score in e_l_dict.items():
            cursor.execute("SELECT text FROM documents WHERE id = ?", (doc_id,))
            text = cursor.fetchone()[0]
            evidence = Evidence(claim, text, score, doc_id)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        answerability_filter(evidence_wrapper, self.answerability_pipe)
        return evidence_wrapper