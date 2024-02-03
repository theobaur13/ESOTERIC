from models import Claim

class ClaimGenerator:
    def __init__(self):
        pass

    def generate_claims(self, text):
        return Claim(text)