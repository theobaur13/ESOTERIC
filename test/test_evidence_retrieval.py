def test_retrieve_evidence_with_valid_claim(evidence_retriever, query):
    claim_text = "The moon is made of cheese."
    query.set_text(claim_text)
    evidence = evidence_retriever.retrieve_evidence(query)
    assert evidence is not None