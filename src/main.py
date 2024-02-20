import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from claim_generation import ClaimGenerator
from evidence_retrieval import EvidenceRetriever
import sqlite3

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')

    input_claim = input("Enter claim: ")

    claim_generator = ClaimGenerator(input_claim)
    evidence_retriever = EvidenceRetriever(data_path)

    evidence_collection = []
    claims = claim_generator.generate_claims()
    for claim in claims.get_claims():
        evidence = evidence_retriever.retrieve_evidence(claim)
        evidence_collection.append(evidence)

    for claim in claims.get_claims():
        print("Extracted question:", claim.text)
    for evidences in evidence_collection:
        print("\n\033[1mClaim: {}\033[0m".format(evidences.claim.text))
        for evidence in evidences.get_evidences():
            evidence.set_wiki_url(sqlite3.connect(os.path.join(data_path, 'wiki-pages.db')))
            print("\nEvidence:", evidence.evidence_sentence)
            print("Score:", str(evidence.score))
            print("Doc ID:", evidence.doc_id)
            print("Wiki URL:", evidence.wiki_url)

if __name__ == '__main__':
    main()