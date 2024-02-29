import os
import sqlite3
import re
import spacy
from models import Evidence, EvidenceWrapper
from evidence_retrieval.tools.document_retrieval import match_search, FAISS_search
from evidence_retrieval.tools.passage_retrieval import answerability_filter, passage_extraction
from evidence_retrieval.tools.NER import extract_entities
from transformers import pipeline

class EvidenceRetriever:
    def __init__(self, data_path):
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'data.db'))
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        # self.answerability_pipe = pipeline("text-classification", model="potsawee/longformer-large-4096-answerable-squad2")

    def retrieve_evidence(self, query):
        evidence = self.retrieve_documents(query)
        evidence = self.retrieve_passages(evidence)
        return evidence

    def retrieve_documents(self, query):
        print("Starting document retrieval for query: '" + str(query) + "'")
        evidence_wrapper = EvidenceWrapper(query)

        entities = extract_entities(self.NER_pipe, query)

        docs = []
        for entity in entities:
            match_docs = match_search(entity, self.connection)
            for doc in match_docs:
                if doc not in docs:
                    docs.append(doc)

        disambiguated_docs = []

        for doc in docs:
            doc_id = doc['doc_id']
            pattern = r'\-LRB\-.+\-RRB\-'

            if re.search(pattern, doc_id):
                disambiguated_docs.append(doc)
                docs = [d for d in docs if d['doc_id'] != doc_id]

        for doc in disambiguated_docs:
            doc_id = doc['doc_id']
    
            pattern = r'\-LRB\-(.+)\-RRB\-'
            info = re.search(pattern, doc_id).group(1)

            nlp_info = self.nlp(info)
            nlp_query = self.nlp(query)
            score = nlp_info.similarity(nlp_query)
            doc['score'] = score

        for doc in docs:
            doc['score'] = 1

        docs = docs + disambiguated_docs
        docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:30]

        cursor = self.connection.cursor()
        for id, doc_id, score in [(doc['id'], doc['doc_id'], doc['score']) for doc in docs]:
            cursor.execute("SELECT text FROM documents WHERE id = ?", (id,))
            text = cursor.fetchone()[0]
            evidence = Evidence(query, text, score, doc_id)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        # answerability_filter(evidence_wrapper, self.answerability_pipe)
        return evidence_wrapper