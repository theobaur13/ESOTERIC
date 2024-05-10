import os
import spacy
from haystack.nodes import DensePassageRetriever
from haystack.nodes import FARMReader
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, DistilBertForSequenceClassification
from models import Evidence, EvidenceWrapper, Sentence
from legacy.final_version.utils.document_retrieval import title_match_search, score_docs, text_match_search, extract_questions, extract_answers, extract_questions, extract_polar_questions
from legacy.final_version.utils.NER import extract_entities
from legacy.final_version.utils.docstore_conversion import listdict_to_docstore, wrapper_to_docstore
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

class EvidenceRetriever:
    def __init__(self, title_match_docs_limit=20, title_match_search_threshold=0, answerability_threshold=0.65, answerability_docs_limit=20, text_match_search_db_limit=1000, reader_threshold=0.7 ,questions=[], use_relevancy_model=True, use_polars=False):
        print ("Initialising evidence retriever")

        self.questions = questions
        self.use_relevancy_model = use_relevancy_model
        self.use_polars = use_polars

        # Setup db connection and NLP models
        load_dotenv()
        self.es = Elasticsearch(hosts=[os.environ.get("ES_HOST_URL")], basic_auth=(os.environ.get("ES_USER"), os.environ.get("ES_PASS")))

        # Set limits and cutoffs for document retrieval
        self.title_match_docs_limit = title_match_docs_limit
        self.text_match_search_db_limit = text_match_search_db_limit
        self.answerability_docs_limit = answerability_docs_limit

        self.title_match_search_threshold = title_match_search_threshold
        self.answerability_threshold = answerability_threshold
        self.reader_threshold = reader_threshold

        # Setup NLP models for document retrieval
        print("Initialising NLP models")
        self.nlp = spacy.load('en_core_web_sm')
        self.NER_model = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        self.question_generation_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", max_length=256)
        self.answer_extraction_pipe = pipeline("text2text-generation", model="vabatista/t5-small-answer-extraction-en")

        # Setup similarity model
        self.sim_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') 

        if self.use_relevancy_model:
            # Setup relevance classification model
            relevance_classification_model_dir = os.path.join(os.path.dirname(__file__), 'models', 'relevancy_classification')
            relevance_classification_model = DistilBertForSequenceClassification.from_pretrained(relevance_classification_model_dir)
            relevance_classification_tokenizer = AutoTokenizer.from_pretrained(relevance_classification_model_dir)
            self.relevance_classification_tokenizer_pipe = pipeline('text-classification', model=relevance_classification_model, tokenizer=relevance_classification_tokenizer)

        print("Evidence retriever initialised")

    def retrieve_evidence(self, claim):
        # Retrieve evidence for a given query
        evidence = self.retrieve_documents(claim)
        evidence = self.retrieve_passages(evidence)
        return evidence

    def retrieve_documents(self, claim):
        print("Starting document retrieval for claim: '" + str(claim) + "'")

        # Extract entities from claim
        print("Extracting entities from claim")
        entities = extract_entities(self.answer_extraction_pipe, self.NER_model, claim)
        print("Entities:", entities)

        # Retrieve documents with exact title match inc. docs with disambiguation in title and score them
        print("Searching for titles containing keywords:", entities)
        title_match_docs = title_match_search(entities, self.es)
        print("Scoring documents")
        title_match_docs = score_docs(title_match_docs, claim, self.nlp)

        # Split docs into title matched and disambiguated docs
        exact_title_matched_docs = [doc for doc in title_match_docs if doc['method'] == "title_match"]
        disambiguated_docs = [doc for doc in title_match_docs if doc['method'] == "disambiguation"]

        # Sort title matched docs by score, taking top N docs or docs above a certain threshold
        disambiguated_docs = sorted(disambiguated_docs, key=lambda x: x['score'], reverse=True)[:self.title_match_docs_limit]
        disambiguated_docs = [doc for doc in disambiguated_docs if doc['score'] > self.title_match_search_threshold]

        # Retrieve X documents where entity is mentioned in the text
        print("Searching for documents containing keywords:", entities)
        textually_matched_docs = text_match_search(entities, self.es, self.text_match_search_db_limit)

        # Generate questions for each answer in the query
        claim_answers = extract_answers(self.answer_extraction_pipe, claim)
        print("Claim answers:", claim_answers)
        # questions = []
        for answer in claim_answers:
            question = extract_questions(self.question_generation_pipe, answer['focal'], claim)
            self.questions.append(question)
            print("Question for answer '" + answer['focal'] + "':", question)

        if self.use_polars:
            # Manually generate polar questions (yes/no questions)
            polar_questions = extract_polar_questions(self.nlp, self.question_generation_pipe, claim)
            for polar_question in polar_questions:
                self.questions.append(polar_question)
                print("Polar question:", polar_question)

        # For doc in both disambiguated and textually matched docs, add to doc store
        doc_store = listdict_to_docstore(disambiguated_docs + textually_matched_docs)

        # Initialise retriever
        print("Initialising DPR")
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

        # Retrieve docs for each question keeping the highest scoring docs
        print("Retrieving documents for each question")
        for question in tqdm(self.questions):
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

        # Add evidence to evidence wrapper
        evidence_wrapper = EvidenceWrapper(claim)
        for id, doc_id, score, method, text, embedding in [(doc['id'], doc['doc_id'], doc['score'], doc['method'], doc['text'], doc['embedding']) for doc in return_docs]:
            evidence = Evidence(query=claim, evidence_text=text, id=id, doc_score=score, doc_id=doc_id, doc_retrieval_method=method, embedding=embedding)
            evidence_wrapper.add_evidence(evidence)

        return evidence_wrapper

    def retrieve_passages(self, evidence_wrapper):
        def get_semantic_sim(self, claim, sentence):
            embeddings = self.sim_model.encode([claim, sentence])
            return util.cos_sim(embeddings[0], embeddings[1]).item()
        
        claim = evidence_wrapper.get_claim()

        if self.use_relevancy_model:
            evidences = evidence_wrapper.get_evidences()
            for evidence in evidences:
                evidence_text = evidence.evidence_text
                doc = self.nlp(evidence_text)

                evidence_sentences = []
                for sentence in doc.sents:
                    sentence = sentence.text
                    input_pair = f"{claim} [SEP] {sentence}"
                    result = self.relevance_classification_tokenizer_pipe(input_pair)
                    label = result[0]['label']
                    relevance_score = result[0]['score']
                    similarity_score = get_semantic_sim(self, claim, sentence)
                    if label == "LABEL_1":
                        evidence_sentence = Sentence(sentence=sentence, score=similarity_score, doc_id=evidence.doc_id)
                        evidence_sentences.append(evidence_sentence)

                for sentence in evidence_sentences:
                    sentence.set_start_end(evidence_text)
                    evidence.add_sentence(sentence)

        else:
            # Retrieve passages using BM25 between the claim and evidence sentences
            print("Retrieving passages using BM25")
            
            evidence_texts = [{"doc_id": evidence.doc_id, "text": evidence.evidence_text} for evidence in evidence_wrapper.get_evidences()]

            # Prepare evidence sentences and mappings
            evidence_sentences = []
            original_sentences = []
            sent_doc_ids_map = {}

            # Clean the claim
            cleaned_claim = [token.text for token in self.nlp(claim) if not token.is_stop and not token.is_punct]

            # Process documents
            print("Processing documents")
            for doc in evidence_texts:
                for sent in self.nlp(doc["text"]).sents:
                    original_sentences.append(sent.text)

                    # Tokenize and clean
                    cleaned = [token.text for token in sent if not token.is_stop and not token.is_punct]
                    evidence_sentences.append(cleaned)
                    sent_doc_ids_map[" ".join(cleaned)] = (doc["doc_id"], sent.text)

            # Create BM25 object and score sentences
            print("Scoring sentences")
            bm25 = BM25Okapi(evidence_sentences)
            scores = bm25.get_scores(cleaned_claim)

            # Prepare ranked sentences with original text
            ranked_sentences = sorted([(evidence_sentences[i], scores[i], original_sentences[i]) for i in range(len(evidence_sentences))], key=lambda x: x[1], reverse=True)

            # Only keep sentences with a score above 0
            ranked_sentences = [sentence for sentence in ranked_sentences if sentence[1] > 0]

            # Only keep the top N sentences
            N = 5
            ranked_sentences = ranked_sentences[:N]

            # Add sentences to evidence
            for cleaned, score, original in ranked_sentences:
                doc_id, original_text = sent_doc_ids_map[" ".join(cleaned)]
                for evidence in evidence_wrapper.get_evidences():
                    if evidence.doc_id == doc_id:
                        sentence = Sentence(sentence=original, score=get_semantic_sim(self, claim, original), doc_id=doc_id, method="BM25")
                        sentence.set_start_end(evidence.evidence_text)
                        evidence.add_sentence(sentence)

            # Retrieve passages using the FARM reader
            doc_store = wrapper_to_docstore(evidence_wrapper)

            # Initialise reader
            reader = FARMReader(
                model_name_or_path="distilbert-base-cased-distilled-squad",
                use_gpu=False,
                context_window_size=250,
                )

            # Retrieve passages for each question
            for question in self.questions:
                print("Retrieving passages for question:", question)
                results = reader.predict(query=question, documents=doc_store, top_k=30)

                for answer in results['answers']:
                    passage = answer.context
                    id = answer.document_ids[0]
                    score = get_semantic_sim(self, claim, passage)

                    if score > self.reader_threshold:
                        evidence = evidence_wrapper.get_evidence_by_id(id)
                        if evidence:
                            sentence = Sentence(sentence=passage, score=score, doc_id=evidence.doc_id, question=question, method="FARM")
                            sentence.set_start_end(evidence.evidence_text)
                            evidence.add_sentence(sentence)
        return evidence_wrapper
    
    def flush_questions(self):
        self.questions = []
        print("Questions flushed")