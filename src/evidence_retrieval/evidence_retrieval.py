import os
import sqlite3
import spacy
import sentence_transformers
from tqdm import tqdm
from transformers import pipeline, LongformerTokenizer, LongformerForSequenceClassification
from models import Evidence, EvidenceWrapper
from evidence_retrieval.tools.document_retrieval import title_match_search, score_docs, text_match_search, extract_focals, extract_questions, calculate_answerability_score_SelfCheckGPT, calculate_answerability_score_tiny
from evidence_retrieval.tools.NER import extract_entities

class EvidenceRetriever:
    def __init__(self, data_path, title_match_docs_limit=20, text_match_search_db_limit=100, text_match_search_k_limit=10, title_match_search_threshold=0, text_match_search_threshold=0, answerability_threshold=0.5):
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
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.FAISS_encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
        self.question_generation_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", max_length=256)
        
        # Setup for faster ranking method
        self.answerability_pipe = pipeline("question-answering", model="deepset/tinyroberta-squad2")

        # # Setup for slower ranking method
        # self.answerability_tokeniser = LongformerTokenizer.from_pretrained("potsawee/longformer-large-4096-answerable-squad2")
        # self.answerability_model = LongformerForSequenceClassification.from_pretrained("potsawee/longformer-large-4096-answerable-squad2")

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
            match_docs = text_match_search(query, entity, self.connection, self.FAISS_encoder, limit=self.text_match_search_db_limit, k_lim=self.text_match_search_k_limit)
            for doc in match_docs:
                if doc not in textually_matched_docs:
                    textually_matched_docs.append(doc)

        # Sort title matched docs by score and take top 20
        docs = sorted(docs, key=lambda x: x['score'], reverse=True)[:self.title_match_docs_limit]

        # Apply threshold to title matched docs
        docs = [doc for doc in docs if doc['score'] > self.title_match_search_threshold]

        # Apply threshold to textually matched docs
        textually_matched_docs = [doc for doc in textually_matched_docs if doc['score'] > self.text_match_search_threshold]

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
            print("Question for focal point '" + focal['entity'] + "':", question)

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

        # Apply threshold to answerability scores
        docs = [doc for doc in docs if doc['score'] > self.answerability_threshold]

        # Retrieve the text of 30 documents from db
        cursor = self.connection.cursor()
        for id, doc_id, score, method, entity in [(doc['id'], doc['doc_id'], doc['score'], doc['method'], doc['entity']) for doc in docs]:
            cursor.execute("SELECT text FROM documents WHERE id = ?", (id,))
            text = cursor.fetchone()[0]
            evidence = Evidence(query=query, evidence_text=text, doc_score=score, doc_id=doc_id, doc_retrieval_method=method, entity=entity)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        evidence_texts = [evidence.evidence_text for evidence in evidence_wrapper.get_evidences()]
        base_claim = evidence_wrapper.get_claim()
        base_claim_focals = extract_focals(self.nlp, base_claim)
        print("Base claim focals:", base_claim_focals)

        questions = []
        for focal in base_claim_focals:
            question = extract_questions(self.question_generation_pipe, focal['entity'], base_claim)
            questions.append(question)
        print("Questions:", questions)

        sentence_ranking = []
        # Extract focal points from the query
        for doc_text in evidence_texts:
            for sentence in self.nlp(doc_text).sents:
                print("Processing sentence:", sentence.text)
                # sentence_focals = extract_focals(self.nlp, sentence.text)
                # print("Sentence focals:", sentence_focals)

                sentence_score = 0
                for question in questions:
                    answerability_score = calculate_answerability_score_tiny(self.answerability_tokeniser, self.answerability_model, sentence.text, question)
                    if answerability_score > sentence_score:
                        sentence_score = answerability_score

                sentence_ranking.append({'sentence': sentence.text, 'score': sentence_score})

                # shared_focals = []
                # shared_focal_types = []
                # for focal in sentence_focals:
                #     if focal in base_claim_focals:
                #         shared_focals.append(focal)
                #     elif focal['type'] in [focal['type'] for focal in base_claim_focals]:
                #         shared_focal_types.append(focal['type'])
                #     elif (focal['type'] == "PERSON"):
                #         name = focal['entity']
                #         name_parts = name.split()

                #         # if two name parts are also found in a base claim focal, then add to shared_focals
                #         for name_claim_focal in [focal['entity'] for focal in base_claim_focals if focal['type'] == "PERSON"]:
                #             if len([part for part in name_parts if part in name_claim_focal.split()]) > 1:
                #                 shared_focals.append(focal)
                #                 break

                # # If sentence contains more than 1 focal point equal to focals in the base claim, sentence score = 1
                # if len(shared_focals) > 1:
                #     sentence_score = 1
                #     sentence_ranking.append({'sentence': sentence.text, 'score': sentence_score})

                # # If sentence contains 1 focal point equal to focals in the base claim and sentence contains a focal type equal to a focal type in the base claim, sentence score = 1
                # elif (len(shared_focals) == 1) and (len(shared_focal_types) > 0):
                #     sentence_score = 1
                #     sentence_ranking.append({'sentence': sentence.text, 'score': sentence_score})
                
                # # If sentence contains 1 focal point then calculate answerability score
                # elif len(shared_focals) == 1:
                #     sentence_score = 0
                #     for question in questions:
                #         answerability_score = calculate_answerability_score(self.answerability_tokeniser, self.answerability_model, sentence.text, question)
                #         if answerability_score > sentence_score:
                #             sentence_score = answerability_score
                #     sentence_ranking.append({'sentence': sentence.text, 'score': sentence_score})
                # # If sentence contains no focal points equal to focals in the base claim, sentence score = 0
                # else:
                #     sentence_score = 0
                #     sentence_ranking.append({'sentence': sentence.text, 'score': sentence_score})
        
        sentence_ranking = sorted(sentence_ranking, key=lambda x: x['score'], reverse=True)
        for line in sentence_ranking:
            print(line)
        return evidence_wrapper