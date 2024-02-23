def test_generate_with_valid_claim(query_generator):
    claim = "The moon is made of cheese."
    query_generator.set_claim(claim)
    result = query_generator.generate_queries()
    assert result is not None