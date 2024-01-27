from models import Claim

class ClaimDetector:
    def __init__(self):
        pass

    def detect_claims(self, text):
        return Claim(text)