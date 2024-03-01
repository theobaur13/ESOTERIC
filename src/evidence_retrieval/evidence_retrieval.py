import os
import sqlite3
import re
import spacy
from models import Evidence, EvidenceWrapper
from evidence_retrieval.tools.document_retrieval import title_match_search, score_docs, text_match_search
from evidence_retrieval.tools.NER import extract_entities
from transformers import pipeline
import sentence_transformers

class EvidenceRetriever:
    def __init__(self, data_path):
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'data.db'))
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")

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
            match_docs = title_match_search(entity, self.connection)
            for doc in match_docs:
                if doc not in docs:
                    docs.append(doc)

        docs = score_docs(docs, query, self.nlp)

        textually_matched_docs = []
        for entity in entities:
            match_docs = text_match_search(query, entity, self.connection, self.encoder)
            for doc in match_docs:
                if doc not in textually_matched_docs:
                    textually_matched_docs.append(doc)

        docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:20]

        for doc in textually_matched_docs:
            if doc not in docs:
                docs.append(doc)

        cursor = self.connection.cursor()
        for id, doc_id, score, method in [(doc['id'], doc['doc_id'], doc['score'], doc['method']) for doc in docs]:
            cursor.execute("SELECT text FROM documents WHERE id = ?", (id,))
            text = cursor.fetchone()[0]
            evidence = Evidence(query, text, score, doc_id, doc_retrieval_method=method, entity=entity)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        return evidence_wrapper