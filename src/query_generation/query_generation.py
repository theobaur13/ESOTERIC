import spacy
from models import Query, QueryWrapper
from transformers import pipeline
from query_generation.tools.varifocal import extract_focals, generate_questions, reranking

class QueryGenerator:
    def __init__(self, claim=""):
        self.text = claim
        self.nlp = spacy.load('en_core_web_sm')
        self.pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

    def set_claim(self, claim):
        self.text = claim

    def generate_queries(self):
        focal_points = extract_focals(self.nlp, self.text)
        questions = generate_questions(self.pipe, self.text, focal_points)

        base_claim = Query(self.text)
        query_wrapper = QueryWrapper(base_claim)

        for question in questions:
            query = Query(question)
            query_wrapper.add_query(query)

        return query_wrapper