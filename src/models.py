class Query:
    def __init__(self, text="", creation_method=""):
        self.text = text
        self.creation_method = creation_method

    def set_text(self, text):
        self.text = text

    def set_creation_method(self, creation_method):
        creation_methods = ["NER", "TE", "VF"]
        if creation_method in creation_methods:
            self.creation_method = creation_method
        else:
            raise ValueError("Invalid creation method. Must be one of: " + ", ".join(creation_methods))

class QueryWrapper:
    def __init__(self, base_claim):
        self.base_claim = base_claim
        self.subqueries = []

    def add_query(self, query):
        self.subqueries.append(query)

    def get_all(self):
        return [self.base_claim] + self.subqueries
    
    def get_base_claim(self):
        return self.base_claim
    
    def get_subqueries(self):
        return self.subqueries

class Evidence:
    def __init__(self, query, evidence_text, score=0, doc_id=None, wiki_url=None, evidence_sentence=None):
        self.query = query
        self.doc_id = doc_id
        self.score = score
        self.evidence_text = evidence_text
        self.wiki_url = wiki_url
        self.evidence_sentence = evidence_sentence

    def set_evidence_sentence(self, evidence_sentence):
        self.evidence_sentence = evidence_sentence

    def set_wiki_url(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT doc_id FROM documents WHERE id = ?", (self.doc_id,))
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
        return self.claim
    
    def remove_evidence(self, evidence):
        self.evidences.remove(evidence)