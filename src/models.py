class Claim:
    def __init__(self, text):
        self.text = text

class ClaimWrapper:
    def __init__(self):
        self.claims = []

    def add_claim(self, claim):
        self.claims.append(claim)

    def get_claims(self):
        return self.claims

class Evidence:
    def __init__(self, claim, evidence_sentence, score=0, doc_id=None, wiki_url=None):
        self.claim = claim
        self.doc_id = doc_id
        self.score = score
        self.evidence_sentence = evidence_sentence
        self.wiki_url = wiki_url

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