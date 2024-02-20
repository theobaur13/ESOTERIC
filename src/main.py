import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from claim_generation import ClaimGenerator
from evidence_retrieval import EvidenceRetriever
from models import EvidenceWrapper

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')

    input_claim = input("Enter claim: ")

    claim_generator = ClaimGenerator(input_claim)
    evidence_retriever = EvidenceRetriever(data_path)

    evidence_collection = EvidenceWrapper()
    claims = claim_generator.generate_claims()
    for claim in claims.get_claims():
        evidence = evidence_retriever.retrieve_evidence(claim)
        for e in evidence.get_evidences():
            evidence_collection.add_evidence(e)

    for claim in claims.get_claims():
        print("Extracted question:", claim.text)
    for evidence in evidence_collection.get_evidences():
        print("\nClaim:", evidence.claim.text)
        print("Evidence:", evidence.evidence_sentence)
        print("Score:", str(evidence.score))
        print("Doc ID:", evidence.doc_id)

if __name__ == '__main__':
    main()