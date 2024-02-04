import os
import unittest

from evidence_retrieval import EvidenceRetriever
from models import Claim

class TestEvidenceRetrieval(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(current_dir, '..', 'data', 'train2.jsonl')
        self.evidence_retriever = EvidenceRetriever(self.dataset_path)

    def test_retrieve_with_valid_claim(self):
        # test with a claim that should return evidence
        claim = Claim("Health care reform legislation is likely to mandate free sex change surgeries.")
        evidence_collection = self.evidence_retriever.retrieve_evidence(claim)
        self.assertTrue(len(evidence_collection.get_evidences()) > 0)
        self.assertTrue(evidence_collection.get_evidences()[0].score > 0.5)
        self.assertTrue(evidence_collection.get_evidences()[0].id == 3)

if __name__ == '__main__':
    unittest.main()