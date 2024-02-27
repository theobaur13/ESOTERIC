import spacy
from models import Query, QueryWrapper
from transformers import pipeline
from query_generation.tools.varifocal import extract_focals, generate_questions, reranking
from query_generation.tools.NER import extract_entities
from query_generation.tools.triple_extraction import extract_triples

class QueryGenerator:
    def __init__(self, claim=""):
        self.text = claim

        # Load NLP models
        self.nlp = spacy.load('en_core_web_sm')
        
        # Load model for question generation
        self.question_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

        # Load model for Named Entity Recognition
        self.NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)

    def set_claim(self, claim):
        self.text = claim

    def generate_queries(self):
        # Extract questions from the claim
        focal_points = extract_focals(self.nlp, self.text)
        questions = generate_questions(self.question_pipe, self.text, focal_points)

        # Extract entities from the claim
        entities = extract_entities(self.NER_pipe, self.text)

        # Extract triples from the claim
        triples = extract_triples(self.nlp, self.text)

        # Construct queries
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

        for triple in triples:
            query = Query(str(triple['subject']) + " " + str(triple['verb']) + " " + str(triple['object']) + " " + str(triple['context']))
            query.set_creation_method("TE")
            query_wrapper.add_query(query)

        return query_wrapper