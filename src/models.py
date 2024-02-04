class Claim:
    def __init__(self, text):
        self.text = text

class Evidence:
    def __init__(self, claim, evidence, score=0, id=None):
        self.id = id
        self.claim = claim
        self.evidence = evidence
        self.score = score

class EvidenceWrapper:
    def __init__(self):
        self.evidences = []

    def add_evidence(self, evidence):
        self.evidences.append(evidence)

    def get_evidences(self):
        return self.evidences