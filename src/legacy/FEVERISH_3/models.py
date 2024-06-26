class Evidence:
    def __init__(self, query, evidence_text, score=0, doc_id=None, wiki_url=None, evidence_sentence=None, doc_retrieval_method=None, entity=None):
        self.query = query
        self.doc_id = doc_id
        self.doc_score = score
        self.evidence_text = evidence_text
        self.evidence_sentence = evidence_sentence
        self.wiki_url = wiki_url
        self.doc_retrieval_method = doc_retrieval_method
        self.entity = entity

    def set_evidence_sentence(self, evidence_sentence):
        self.evidence_sentence = evidence_sentence

    def set_wiki_url(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (self.doc_id,))
        self.wiki_url = "https://en.wikipedia.org/wiki/" + str(cursor.fetchone()[0])

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