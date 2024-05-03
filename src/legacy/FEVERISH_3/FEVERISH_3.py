import os
import sqlite3
import spacy
from legacy.FEVERISH_3.models import Evidence, EvidenceWrapper
from legacy.FEVERISH_3.utils.NER import extract_entities
from legacy.FEVERISH_3.utils.document_retrieval import title_match_search, score_docs, text_match_search
from transformers import pipeline
import sentence_transformers

class EvidenceRetriever:
    def __init__(self, data_path):
        # Set up db connection and NLP models
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'data.db'))
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")

    def retrieve_evidence(self, query):
        # Retrieve evidence for a given query
        evidence = self.retrieve_documents(query)
        evidence = self.retrieve_passages(evidence)
        return evidence

    def retrieve_documents(self, query):
        print("Starting document retrieval for query: '" + str(query) + "'")
        evidence_wrapper = EvidenceWrapper(query)

        entities = extract_entities(self.NER_pipe, query)

        # Retrieve documents with exact title match inc. docs with disambiguation in title
        docs = []
        for entity in entities:
            match_docs = title_match_search(entity, self.connection)
            for doc in match_docs:
                if doc not in docs:
                    docs.append(doc)

        # Rerank documents:
        # score = 1 for exact match
        # score = cosine_similarity(info, query) for disambiguated docs
        docs = score_docs(docs, query, self.nlp)

        # Retrieve 100 documents where entity is mentioned in the text
        # Rerank according to FAISS inner product between query and text
        # Return sorted top 10 documents
        textually_matched_docs = []
        for entity in entities:
            match_docs = text_match_search(query, entity, self.connection, self.encoder)
            for doc in match_docs:
                if doc not in textually_matched_docs:
                    textually_matched_docs.append(doc)

        # Sort textually matched docs and take top 10
        textually_matched_docs = sorted(textually_matched_docs, key=lambda x: x['score'], reverse=True)[:10]

        # Sort title matched docs by score and take top 10
        docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:10]

        # Add textually matched docs to the list
        for doc in textually_matched_docs:
            if doc not in docs:
                docs.append(doc)

        # Retrieve the text of 30 documents from db
        cursor = self.connection.cursor()
        for id, doc_id, score, method, entity in [(doc['id'], doc['doc_id'], doc['score'], doc['method'], doc['entity']) for doc in docs]:
            if doc_id not in [e.doc_id for e in evidence_wrapper.get_evidences()]:
                cursor.execute("SELECT text FROM documents WHERE id = ?", (id,))
                text = cursor.fetchone()[0]
                evidence = Evidence(query, text, score, doc_id, doc_retrieval_method=method, entity=entity)
                evidence_wrapper.add_evidence(evidence)
        
        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        return evidence_wrapper