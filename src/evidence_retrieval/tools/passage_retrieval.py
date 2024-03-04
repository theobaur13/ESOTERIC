def passage_extraction(evidence_collection, nlp):
    base_claim = evidence_collection.get_claim()
    nlp_base_claim = nlp(base_claim)

    for evidence in evidence_collection.get_evidences():
        text = evidence.evidence_text
        sents = []
        
        for sentence in nlp(text).sents:
            nlp_sentence = nlp(sentence.text)
            similarity = nlp_sentence.similarity(nlp_base_claim)
            sents.append({'sentence': sentence.text, 'score': similarity})

        sents = sorted(sents, key=lambda x: x['score'], reverse=True)
        evidence.set_evidence_sentence(sents[0]['sentence'])
        evidence.set_evidence_score(sents[0]['score'])

    return evidence_collection