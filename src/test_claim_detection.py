import sys
import os
import unittest

from claim_detection import ClaimDetector

class TestClaimDetection(unittest.TestCase):
    def setUp(self):
        self.claim_detector = ClaimDetector()

    def test_detect_with_valid_claim(self):
        # test with a claim that should return true
        claim = self.claim_detector.detect_claims("The Earth is round")
        self.assertIsNotNone(claim)

if __name__ == '__main__':
    unittest.main()