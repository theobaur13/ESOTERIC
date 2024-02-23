def test_retrieve_evidence_with_valid_claim(evidence_retriever, claim):
    claim_text = "The moon is made of cheese."
    claim.set_text(claim_text)
    evidence = evidence_retriever.retrieve_evidence(claim)
    assert evidence is not None