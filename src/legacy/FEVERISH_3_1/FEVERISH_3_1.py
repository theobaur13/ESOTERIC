import os
import sqlite3
import spacy
import sentence_transformers
from tqdm import tqdm
from transformers import pipeline
from legacy.FEVERISH_3_1.models import Evidence, EvidenceWrapper
from legacy.FEVERISH_3_1.utils.document_retrieval import title_match_search, score_docs, text_match_search, extract_focals, extract_questions, calculate_answerability_score_SelfCheckGPT, calculate_answerability_score_tiny
from legacy.FEVERISH_3_1.utils.NER import extract_entities

class EvidenceRetriever:
    def __init__(self, data_path):
        # Setup db connection and NLP models
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'data.db'))

        # Setup NLP models for document retrieval
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.FAISS_encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.question_generation_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", max_length=256)

        # Setup for faster ranking method
        self.answerability_pipe = pipeline("question-answering", model="deepset/tinyroberta-squad2")

    def retrieve_evidence(self, query):
        # Retrieve evidence for a given query
        evidence = self.retrieve_documents(query)
        # evidence = self.retrieve_passages(evidence)
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
            match_docs = text_match_search(query, entity, self.connection, self.FAISS_encoder)
            for doc in match_docs:
                if doc not in textually_matched_docs:
                    textually_matched_docs.append(doc)

        # Sort title matched docs by score and take top 20
        docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:20]

        # Add textually matched docs to the list
        for doc in textually_matched_docs:
            if doc not in docs:
                docs.append(doc)

        # Generate questions for each focal point in the query
        claim_focals = extract_focals(self.nlp, query)
        print("Claim focals:", claim_focals)

        questions = []
        for focal in claim_focals:
            question = extract_questions(self.question_generation_pipe, focal['entity'], query)
            questions.append(question)

        # Rescore documents by answerability
        for doc in tqdm(docs):
            doc_id = doc['doc_id']
            cursor = self.connection.cursor()
            cursor.execute("SELECT text FROM documents WHERE doc_id = ?", (doc_id,))
            text = cursor.fetchone()[0]

            doc_score = 0
            for question in questions:
                # answerability_score = calculate_answerability_score_SelfCheckGPT(self.answerability_tokeniser, self.answerability_model, text, question)
                answerability_score = calculate_answerability_score_tiny(self.answerability_pipe, text, question)
                if answerability_score > doc_score:
                    doc_score = answerability_score

            doc['score'] = doc_score

        # Retrieve the text of 30 documents from db
        cursor = self.connection.cursor()
        for id, doc_id, score, method, entity in [(doc['id'], doc['doc_id'], doc['score'], doc['method'], doc['entity']) for doc in docs]:
            cursor.execute("SELECT text FROM documents WHERE id = ?", (id,))
            text = cursor.fetchone()[0]
            evidence = Evidence(query=query, evidence_text=text, doc_score=score, doc_id=doc_id, doc_retrieval_method=method, entity=entity)
            evidence_wrapper.add_evidence(evidence)

        evidence_wrapper.sort_by_doc_score()
        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        return evidence_wrapper