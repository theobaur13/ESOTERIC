class Evidence:
    def __init__(self, query, evidence_text, doc_id=None, doc_score=0, sentences=[], wiki_url=None, doc_retrieval_method=None, entity=None):
        self.query = query

        self.doc_id = doc_id
        self.doc_score = doc_score
        self.evidence_text = evidence_text
        self.sentences = sentences

        self.wiki_url = wiki_url
        self.doc_retrieval_method = doc_retrieval_method
        self.entity = entity

    def set_evidence_sentences(self, sentences):
        self.sentences = sentences

    def set_wiki_url(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (self.doc_id,))
        self.wiki_url = "https://en.wikipedia.org/wiki/" + str(cursor.fetchone()[0])

    def __str__(self):
        return f"Query: {self.query}\nDoc ID: {self.doc_id}\nDoc Score: {self.doc_score}\nEvidence Text: {self.evidence_text}\nSentences: {self.sentences}\nWiki URL: {self.wiki_url}\nDoc Retrieval Method: {self.doc_retrieval_method}\nEntity: {self.entity}"

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