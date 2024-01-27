import os
from claim_detection import ClaimDetector
from evidence_retrieval import EvidenceRetriever

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'data', 'dataset.json')

    claim_detector = ClaimDetector()
    evidence_retriever = EvidenceRetriever(dataset_path)

    input_claim = input("Enter claim: ")

    claim = claim_detector.detect_claims(input_claim)
    evidence = evidence_retriever.retrieve_evidence(claim)

    print("Claim:" + claim.text)
    print(evidence.claim.text + " is supported by a " + evidence.evidence[0]["type"] + " concluding that " + evidence.evidence[0]["conclusion"])

if __name__ == '__main__':
    main()