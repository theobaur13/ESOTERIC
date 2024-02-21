import spacy
from models import Claim, ClaimWrapper
from transformers import pipeline

class ClaimGenerator:
    def __init__(self, claim):
        self.text = claim
        self.nlp = spacy.load('en_core_web_sm')
        self.pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

    def generate_claims(self):
        focal_points = self.extract_focals()
        questions = self.generate_questions(focal_points)

        base_claim = Claim(self.text)
        claim_wrapper = ClaimWrapper(base_claim)

        for question in questions:
            claim = Claim(question)
            claim_wrapper.add_claim(claim)

        return claim_wrapper

    def extract_focals(self):
        print("Extracting focal points from claim")
        doc = self.nlp(self.text)
        sentences = doc.sents

        tag_set = {
            "clause_relations": ["csubj", "xcomp", "ccomp", "advcl", "acl"],
            "sentence_core": ["nsubj", "obj", "iobj"],
            "full_syntax": ["csubj", "xcomp", "ccomp", "advcl", "acl", "nsubj", "obj", "iobj"],
            "noun_modifiers": ["nmod", "amod", "obj", "num"],
            "phrase_details": ["nsubj", "obj", "iobj", "nmod", "amod", "obj", "num"],
            "syntax_all": ["csubj", "xcomp", "ccomp", "advcl", "acl", "nsubj", "obj", "iobj", "nmod", "amod", "obj", "num"],
            "sub_verb_obj": ["agent", "ccomp", "csubj", "dobj", "nsubj", "nsubjpass", "pcomp", "pobj", "root"]
        }

        active_tag_set = tag_set["sub_verb_obj"]
        subtrees = []

        for sentence in sentences:
            for token in sentence:
                if token.dep_ in active_tag_set:
                    subtree = [t.text for t in token.subtree]
                    subtrees.append([" ".join(subtree), token.dep_])

        print("Focal points:", subtrees, "extracted.")
        return subtrees
    
    def generate_questions(self, focal_points):
        print("Generating questions from focal points")
        questions = []

        for focal_point in focal_points:
            focal_text = focal_point[0]
            pipe_string = "answer: " + str(focal_text) + " context: " + str(self.text)

            output = self.pipe(pipe_string)
            
            question = output[0]['generated_text'].split("question: ")[1]
            questions.append(question)
        
        print("Questions generated:", questions)
        return questions
    
    def reranking(self, questions):
        print("Reranking questions")
        return questions