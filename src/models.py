class Claim:
    def __init__(self, text):
        self.text = text

class Evidence:
    def __init__(self, claim, evidence):
        self.claim = claim
        self.evidence = evidence