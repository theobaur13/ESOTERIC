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
        print("Claim:", claim.text)
    for evidence in evidence_collection.get_evidences():
        print("Doc ID:",  evidence.doc_id, "Score:", str(evidence.score), "Evidence:", evidence.evidence_sentence)

if __name__ == '__main__':
    main()