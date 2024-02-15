class Claim:
    def __init__(self, text):
        self.text = text

class Evidence:
    def __init__(self, claim, evidence_sentence, score=0, doc_id=None):
        self.claim = claim
        self.doc_id = doc_id
        self.score = score
        self.evidence_sentence = evidence_sentence

class EvidenceWrapper:
    def __init__(self):
        self.evidences = []

    def add_evidence(self, evidence):
        self.evidences.append(evidence)

    def get_evidences(self):
        return self.evidences