class Evidence:
    def __init__(self, query, evidence_text, doc_id=None, doc_score=0, sentences=[], wiki_url=None, doc_retrieval_method=None):
        self.query = query

        self.doc_id = doc_id
        self.doc_score = doc_score
        self.evidence_text = evidence_text
        self.sentences = sentences

        self.doc_retrieval_method = doc_retrieval_method

    def set_evidence_sentences(self, sentences):
        self.sentences = sentences

    def __str__(self):
        return f"Query: {self.query}\nDoc ID: {self.doc_id}\nDoc Score: {self.doc_score}\nEvidence Text: {self.evidence_text}\nSentences: {self.sentences}\nDoc Retrieval Method: {self.doc_retrieval_method}"

class EvidenceWrapper:
    def __init__(self, query):
        self.query = query
        self.evidences = []

    def add_evidence(self, evidence):
        self.evidences.append(evidence)

    def get_evidences(self):
        return self.evidences
    
    def get_claim(self):
        return self.query
    
    def remove_evidence(self, evidence):
        self.evidences.remove(evidence)

    def sort_by_doc_score(self):
        self.evidences = sorted(self.evidences, key=lambda x: x.doc_score, reverse=True)

    def sort_by_sentence_score(self):
        self.evidences = sorted(self.evidences, key=lambda x: x.sentence_score, reverse=True)

        def __str__(self):
            return f"Query: {self.query}\nEvidences: {self.evidences}"

class Sentence:
    def __init__(self, sentence=None, score=0, doc_id=None, sent_id=None):
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.sentence = sentence
        self.score = score

    def __str__(self):
        return f"Doc ID: {self.doc_id}\nSentence ID: {self.sent_id}\nSentence: {self.sentence}\nScore: {self.score}"