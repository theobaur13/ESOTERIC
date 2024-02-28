def test_retrieve_evidence_with_valid_claim(evidence_retriever):
    claim_text = "The moon is made of cheese."
    evidence = evidence_retriever.retrieve_evidence(claim_text)
    assert evidence is not None