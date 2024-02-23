def test_generate_with_valid_claim(claim_generator):
    claim = "The moon is made of cheese."
    claim_generator.set_claim(claim)
    result = claim_generator.generate_claims()
    assert result is not None