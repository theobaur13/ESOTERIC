class Claim:
    def __init__(self, text=""):
        self.text = text

    def set_text(self, text):
        self.text = text

class ClaimWrapper:
    def __init__(self, base_claim):
        self.base_claim = base_claim
        self.subclaims = []

    def add_claim(self, claim):
        self.subclaims.append(claim)

    def get_claims(self):
        return [self.base_claim] + self.subclaims
    
    def get_base_claim(self):
        return self.base_claim
    
    def get_subclaims(self):
        return self.subclaims

class Evidence:
    def __init__(self, claim, evidence_text, score=0, doc_id=None, wiki_url=None, evidence_sentence=None):
        self.claim = claim
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
    def __init__(self, claim):
        self.claim = claim
        self.evidences = []

    def add_evidence(self, evidence):
        self.evidences.append(evidence)

    def get_evidences(self):
        return self.evidences
    
    def get_claim(self):
        return self.claim
    
    def remove_evidence(self, evidence):
        self.evidences.remove(evidence)