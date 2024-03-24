import os
import re
import spacy
import sentence_transformers
from tqdm import tqdm
from transformers import pipeline, DistilBertForSequenceClassification, AutoTokenizer
from models import Evidence, EvidenceWrapper, Sentence
from evidence_retrieval.tools.document_retrieval import title_match_search, score_docs, text_match_search, extract_focals, extract_questions, calculate_answerability_score_SelfCheckGPT, calculate_answerability_score_tiny, extract_answers, extract_questions, extract_polar_questions
from evidence_retrieval.tools.NER import extract_entities
from elasticsearch import Elasticsearch

class EvidenceRetriever:
    def __init__(self, data_path, title_match_docs_limit=20, text_match_search_db_limit=100, text_match_search_k_limit=10, title_match_search_threshold=0, text_match_search_threshold=0, answerability_threshold=0.5):
        print ("Initialising evidence retriever")

        # Setup db connection and NLP models
        self.data_path = data_path
        self.es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "password"))

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
        # self.NER_model = SpanMarkerModel.from_pretrained("lxyuan/span-marker-bert-base-multilingual-uncased-multinerd")
        self.NER_model = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.FAISS_encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.question_generation_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", max_length=256)
        self.answer_extraction_pipe = pipeline("text2text-generation", model="vabatista/t5-small-answer-extraction-en")

        # Setup for faster ranking method
        self.answerability_pipe = pipeline("question-answering", model="deepset/tinyroberta-squad2")

        # Setup relevance classification model
        relevance_classification_model_dir = os.path.join(self.data_path, '..', 'models', 'relevancy_classification')
        relevance_classification_model = DistilBertForSequenceClassification.from_pretrained(relevance_classification_model_dir)
        relevance_classification_tokenizer = AutoTokenizer.from_pretrained(relevance_classification_model_dir)
        self.relevance_classification_tokenizer_pipe = pipeline('text-classification', model=relevance_classification_model, tokenizer=relevance_classification_tokenizer)
        print("Evidence retriever initialised")

    def retrieve_evidence(self, query):
        # Retrieve evidence for a given query
        evidence = self.retrieve_documents(query)
        evidence = self.retrieve_passages(evidence)
        return evidence

    def retrieve_documents(self, query):
        print("Starting document retrieval for query: '" + str(query) + "'")
        evidence_wrapper = EvidenceWrapper(query)

        print("Extracting entities from text")
        entities = extract_entities(self.answer_extraction_pipe, self.NER_model, query)
        print("Entities:", entities)

        # Retrieve documents with exact title match inc. docs with disambiguation in title
        title_match_docs = title_match_search(entities, self.es)

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
        textually_matched_docs = text_match_search(query, entities, self.es, self.FAISS_encoder, self.text_match_search_db_limit, self.text_match_search_k_limit)

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
            id = doc['id']
            text = doc['text']

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
        for id, doc_id, score, method, text in [(doc['id'], doc['doc_id'], doc['score'], doc['method'], doc['text']) for doc in docs]:
            evidence = Evidence(query=query, evidence_text=text, doc_score=score, doc_id=doc_id, doc_retrieval_method=method)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        base_claim = evidence_wrapper.get_claim()
        evidences = evidence_wrapper.get_evidences()
        for evidence in evidences:
            text = evidence.evidence_text

            pattern = r'\n\d+\t'
            sentences = re.split(pattern, text)
            sentences = [sentence for sentence in sentences if sentence != ""]

            evidence_sentences = []
            for sentence in sentences:
                sent_id = sentences.index(sentence)
                input_pair = f"{base_claim} [SEP] {sentence}"
                result = self.relevance_classification_tokenizer_pipe(input_pair)
                label = result[0]['label']
                relevance_score = result[0]['score']
                if label == "LABEL_1":
                    evidence_sentence = Sentence(sentence=sentence, score=relevance_score, doc_id=evidence.doc_id, sent_id=sent_id)
                    evidence_sentences.append(evidence_sentence)

            evidence.set_evidence_sentences(evidence_sentences)
        return evidence_wrapper