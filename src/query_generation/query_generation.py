import spacy
from models import Query, QueryWrapper
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from query_generation.tools.varifocal import extract_focals, generate_questions, reranking
from query_generation.tools.NER import extract_entities

class QueryGenerator:
    def __init__(self, claim=""):
        self.text = claim

        # Load NLP models
        self.nlp = spacy.load('en_core_web_sm')
        self.question_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)

    def set_claim(self, claim):
        self.text = claim

    def generate_queries(self):
        focal_points = extract_focals(self.nlp, self.text)
        questions = generate_questions(self.question_pipe, self.text, focal_points)

        entities = extract_entities(self.NER_pipe, self.text)

        base_claim = Query(self.text)
        query_wrapper = QueryWrapper(base_claim)

        for question in questions:
            query = Query(question)
            query.set_creation_method("VF")
            query_wrapper.add_query(query)

        for entity in entities:
            query = Query(entity)
            query.set_creation_method("NER")
            query_wrapper.add_query(query)

        return query_wrapper