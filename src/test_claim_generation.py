import unittest

from claim_generation import ClaimGenerator

class TestClaimDetection(unittest.TestCase):
    def setUp(self):
        self.claim_generator = ClaimGenerator()

    def test_generate_with_valid_claim(self):
        # test with a claim that should return claims
        claim = self.claim_generator.generate_claims("The Earth is round")
        self.assertIsNotNone(claim)

if __name__ == '__main__':
    unittest.main()