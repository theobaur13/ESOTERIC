import os
import re
import spacy
import sentence_transformers
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import DensePassageRetriever
from tqdm import tqdm
from transformers import pipeline, DistilBertForSequenceClassification, AutoTokenizer
from models import Evidence, EvidenceWrapper, Sentence
from evidence_retrieval.tools.document_retrieval import title_match_search, score_docs, text_match_search, extract_focals, extract_questions, calculate_answerability_score_SelfCheckGPT, calculate_answerability_score_tiny, extract_answers, extract_questions, extract_polar_questions
from evidence_retrieval.tools.NER import extract_entities
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

class EvidenceRetriever:
    def __init__(self, data_path, title_match_docs_limit=20, title_match_search_threshold=0, answerability_threshold=0.5, answerability_docs_limit=20, text_match_search_db_limit=1000):
        print ("Initialising evidence retriever")

        # Setup db connection and NLP models
        load_dotenv()
        self.data_path = data_path
        self.es = Elasticsearch(hosts=[os.environ.get("ES_HOST_URL")], basic_auth=(os.environ.get("ES_USER"), os.environ.get("ES_PASS")))

        # Set limits and cutoffs for document retrieval
        self.title_match_docs_limit = title_match_docs_limit
        self.text_match_search_db_limit = text_match_search_db_limit
        self.answerability_docs_limit = answerability_docs_limit

        self.title_match_search_threshold = title_match_search_threshold
        self.answerability_threshold = answerability_threshold

        # Setup NLP models for document retrieval
        print("Initialising NLP models")
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_model = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.question_generation_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", max_length=256)
        self.answer_extraction_pipe = pipeline("text2text-generation", model="vabatista/t5-small-answer-extraction-en")

        # Setup relevance classification model
        relevance_classification_model_dir = os.path.join(self.data_path, '..', 'models', 'relevancy_classification')
        relevance_classification_model = DistilBertForSequenceClassification.from_pretrained(relevance_classification_model_dir)
        relevance_classification_tokenizer = AutoTokenizer.from_pretrained(relevance_classification_model_dir)
        self.relevance_classification_tokenizer_pipe = pipeline('text-classification', model=relevance_classification_model, tokenizer=relevance_classification_tokenizer)
        print("Evidence retriever initialised")

    def retrieve_evidence(self, claim):
        # Retrieve evidence for a given query
        evidence = self.retrieve_documents(claim)
        # evidence = self.retrieve_passages(evidence)
        return evidence

    def retrieve_documents(self, claim):
        print("Starting document retrieval for query: '" + str(claim) + "'")
        evidence_wrapper = EvidenceWrapper(claim)

        print("Extracting entities from text")
        entities = extract_entities(self.answer_extraction_pipe, self.NER_model, claim)
        print("Entities:", entities)

        # Retrieve documents with exact title match inc. docs with disambiguation in title
        title_match_docs = title_match_search(entities, self.es)

        # Rerank documents:
        # score = 1 for exact match
        # score = cosine_similarity(info, query) for disambiguated docs
        title_match_docs = score_docs(title_match_docs, claim, self.nlp)
    
        # Split docs into title matched and disambiguated docs
        exact_title_matched_docs = [doc for doc in title_match_docs if doc['method'] == "title_match"]
        disambiguated_docs = [doc for doc in title_match_docs if doc['method'] == "disambiguation"]

        # Sort title matched docs by score and take top 20
        disambiguated_docs = sorted(disambiguated_docs, key=lambda x: x['score'], reverse=True)[:self.title_match_docs_limit]

        # Apply threshold to title matched docs
        disambiguated_docs = [doc for doc in disambiguated_docs if doc['score'] > self.title_match_search_threshold]

        # Retrieve X documents where entity is mentioned in the text
        textually_matched_docs = text_match_search(entities, self.es, self.text_match_search_db_limit)

        # Generate questions for each focal point in the query
        claim_answers = extract_answers(self.answer_extraction_pipe, claim)
        print("Claim answers:", claim_answers)

        questions = []
        for answer in claim_answers:
            question = extract_questions(self.question_generation_pipe, answer['focal'], claim)
            questions.append(question)
            print("Question for focal point '" + answer['focal'] + "':", question)

        # Manually add polar questions for each focal point
        polar_questions = extract_polar_questions(self.nlp, self.question_generation_pipe, claim)
        for polar_question in polar_questions:
            questions.append(polar_question)
            print("Polar question:", polar_question)

        # For doc in both disambiguated and textually matched docs, add to doc store
        doc_store = InMemoryDocumentStore()
        for doc in disambiguated_docs + textually_matched_docs:
            id = doc['id']
            doc_id = doc['doc_id']
            content = doc['text']
            content_type = "text"
            embedding = doc['embedding']
            meta = {"doc_id": doc_id}
            
            # if not already in doc store, add to doc store
            if id not in [d.id for d in doc_store.get_all_documents()]:
                doc = Document(id=id, doc_id=doc_id, content=content, content_type=content_type, embedding=embedding, meta=meta)
                doc_store.write_documents([doc])

        retriever = DensePassageRetriever(
            document_store=doc_store,
            query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
            passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
            use_gpu=False,
            embed_title=True,
            batch_size=2,
        )

        # Set docs to return
        return_docs = []
        for doc in exact_title_matched_docs:
            return_docs.append(doc)

        for question in questions:
            results = retriever.retrieve(query=question)
            for result in results:
                id = result.id
                score = result.score

                if score > self.answerability_threshold:
                    for doc in disambiguated_docs + textually_matched_docs:
                        if doc['id'] == id and doc['score'] < score:
                            if doc["id"] not in [d["id"] for d in return_docs]:
                                doc['score'] = score
                                return_docs.append(doc)

        for id, doc_id, score, method, text in [(doc['id'], doc['doc_id'], doc['score'], doc['method'], doc['text']) for doc in return_docs]:
            evidence = Evidence(query=claim, evidence_text=text, doc_score=score, doc_id=doc_id, doc_retrieval_method=method)
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