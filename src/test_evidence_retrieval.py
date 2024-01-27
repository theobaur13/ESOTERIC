import sys
import os
import unittest

from evidence_retrieval import EvidenceRetriever
from models import Claim

class TestEvidenceRetrieval(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(current_dir, '..', 'data', 'dataset.json')
        self.evidence_retriever = EvidenceRetriever(self.dataset_path)

    def test_retrieve_with_valid_claim(self):
        # test with a claim that should return evidence
        claim = Claim("The Earth is round")
        evidence = self.evidence_retriever.retrieve_evidence(claim)
        
        self.assertTrue(len(evidence.evidence) > 0)

if __name__ == '__main__':
    unittest.main()