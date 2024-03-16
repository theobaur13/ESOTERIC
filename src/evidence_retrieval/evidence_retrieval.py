import os
import sqlite3
import spacy
import sentence_transformers
from tqdm import tqdm
from transformers import pipeline, LongformerTokenizer, LongformerForSequenceClassification
from models import Evidence, EvidenceWrapper
from evidence_retrieval.tools.document_retrieval import title_match_search, score_docs, text_match_search, extract_focals, extract_questions, calculate_answerability_score_SelfCheckGPT, calculate_answerability_score_tiny, extract_answers, extract_questions, extract_polar_questions
from evidence_retrieval.tools.NER import extract_entities

class EvidenceRetriever:
    def __init__(self, data_path, title_match_docs_limit=20, text_match_search_db_limit=100, text_match_search_k_limit=10, title_match_search_threshold=0, text_match_search_threshold=0, answerability_threshold=0.5):
        print ("Initialising evidence retriever")

        # Setup db connection and NLP models
        self.data_path = data_path
        self.connection = sqlite3.connect(os.path.join(self.data_path, 'data.db'))

        # Set limits and cutoffs for document retrieval
        self.title_match_docs_limit = title_match_docs_limit
        self.text_match_search_db_limit = text_match_search_db_limit
        self.text_match_search_k_limit = text_match_search_k_limit

        self.title_match_search_threshold = title_match_search_threshold
        self.text_match_search_threshold = text_match_search_threshold
        self.answerability_threshold = answerability_threshold

        # Setup NLP models for document retrieval
        print("Initialising NLP models")
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.FAISS_encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.question_generation_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", max_length=256)
        self.answer_extraction_pipe = pipeline("text2text-generation", model="vabatista/t5-small-answer-extraction-en")

        # Setup for faster ranking method
        self.answerability_pipe = pipeline("question-answering", model="deepset/tinyroberta-squad2")

        print("Evidence retriever initialised")

    def retrieve_evidence(self, query):
        # Retrieve evidence for a given query
        evidence = self.retrieve_documents(query)
        evidence = self.retrieve_passages(evidence)
        return evidence

    def retrieve_documents(self, query):
        print("Starting document retrieval for query: '" + str(query) + "'")
        evidence_wrapper = EvidenceWrapper(query)

        entities = extract_entities(self.answer_extraction_pipe, self.NER_pipe, query)

        # Retrieve documents with exact title match inc. docs with disambiguation in title
        title_match_docs = []
        for entity in entities:
            match_docs = title_match_search(entity, self.connection)
            for doc in match_docs:
                if doc not in title_match_docs:
                    title_match_docs.append(doc)

        # Rerank documents:
        # score = 1 for exact match
        # score = cosine_similarity(info, query) for disambiguated docs
        title_match_docs = score_docs(title_match_docs, query, self.nlp)
    
        # Split docs into title matched and disambiguated docs
        exact_title_matched_docs = [doc for doc in title_match_docs if doc['method'] == "title_match"]
        disambiguated_docs = [doc for doc in title_match_docs if doc['method'] == "disambiguation"]

        # Sort title matched docs by score and take top 20
        disambiguated_docs = sorted(disambiguated_docs, key=lambda x: x['score'], reverse=True)[:self.title_match_docs_limit]

        # Apply threshold to title matched docs
        disambiguated_docs = [doc for doc in disambiguated_docs if doc['score'] > self.title_match_search_threshold]

        # Retrieve X documents where entity is mentioned in the text
        # Rerank according to FAISS inner product between query and text
        # Return sorted top Y documents
        textually_matched_docs = []
        for entity in entities:
            match_docs = text_match_search(query, entity, self.connection, self.FAISS_encoder, limit=self.text_match_search_db_limit, k_lim=self.text_match_search_k_limit)
            for doc in match_docs:
                if doc not in textually_matched_docs:
                    textually_matched_docs.append(doc)

        # Apply threshold to textually matched docs
        textually_matched_docs = [doc for doc in textually_matched_docs if doc['score'] > self.text_match_search_threshold]

        # Generate questions for each focal point in the query
        claim_focals = extract_answers(self.answer_extraction_pipe, query)
        print("Claim focals:", claim_focals)

        questions = []
        for focal in claim_focals:
            question = extract_questions(self.question_generation_pipe, focal['focal'], query)
            questions.append(question)
            print("Question for focal point '" + focal['focal'] + "':", question)

        # Manually add polar questions for each focal point
        polar_questions = extract_polar_questions(self.nlp, self.question_generation_pipe, query)
        for polar_question in polar_questions:
            questions.append(polar_question)
            print("Polar question:", polar_question)

        # Add disambiguated docs and textually matched docs to list for reranking
        rerank_docs = []
        for doc in disambiguated_docs:
            rerank_docs.append(doc)
        for doc in textually_matched_docs:
            doc_id = doc['doc_id']
            if doc_id not in [d['doc_id'] for d in rerank_docs]:
                rerank_docs.append(doc)

        # Rescore documents by answerability
        for doc in tqdm(rerank_docs):
            doc_id = doc['doc_id']
            cursor = self.connection.cursor()
            cursor.execute("SELECT text FROM documents WHERE doc_id = ?", (doc_id,))
            text = cursor.fetchone()[0]

            doc_score = 0
            for question in questions:
                if text != "":
                    answerability_score = calculate_answerability_score_tiny(self.answerability_pipe, text, question)
                    if answerability_score > doc_score:
                        doc_score = answerability_score

            doc['score'] = doc_score
            
        # Apply threshold to answerability scores
        rerank_docs = [doc for doc in rerank_docs if doc['score'] > self.answerability_threshold]

        # Combine title matched, disambiguated and textually matched docs
        docs = []
        for doc in exact_title_matched_docs:
            docs.append(doc)
        for doc in rerank_docs:
            if doc['doc_id'] not in [d['doc_id'] for d in docs]:
                docs.append(doc)

        # Retrieve the text of 30 documents from db
        cursor = self.connection.cursor()
        for id, doc_id, score, method, entity in [(doc['id'], doc['doc_id'], doc['score'], doc['method'], doc['entity']) for doc in docs]:
            cursor.execute("SELECT text FROM documents WHERE id = ?", (id,))
            text = cursor.fetchone()[0]
            evidence = Evidence(query=query, evidence_text=text, doc_score=score, doc_id=doc_id, doc_retrieval_method=method, entity=entity)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        return evidence_wrapper