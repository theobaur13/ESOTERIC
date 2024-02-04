import os
from claim_generation import ClaimGenerator
from evidence_retrieval import EvidenceRetriever

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'data', 'train2.jsonl')

    claim_detector = ClaimGenerator()
    evidence_retriever = EvidenceRetriever(dataset_path)

    input_claim = input("Enter claim: ")

    claim = claim_detector.generate_claims(input_claim)
    evidence_collection = evidence_retriever.retrieve_evidence(claim)

    print("Claim: " + claim.text)
    print("Evidence:")
    for evidence in evidence_collection.get_evidences():
        print(evidence.evidence + " - " + str(evidence.score) + " - " + str(evidence.id))

if __name__ == '__main__':
    main()