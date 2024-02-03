class Claim:
    def __init__(self, text):
        self.text = text

class Evidence:
    def __init__(self, claim, evidence, score=0):
        self.claim = claim
        self.evidence = evidence
        self.score = score