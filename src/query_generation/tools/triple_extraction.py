def extract_triples(nlp, text):
    print("Extracting triples from text:", text)

    doc = nlp(text)
    sentences = doc.sents

    triples = []

    for sentence in sentences:
        sentence._.clauses
        for clause in sentence._.clauses:
            propositions = clause.to_propositions(as_text=False, inflect=None)
            for proposition in propositions:
                print(proposition)
                subject = proposition[0].text
                verb = ""
                obj = ""
                context = ""
                
                if len(proposition) == 2:
                    verb = str(proposition[1])
                elif len(proposition) == 3:
                    verb = str(proposition[1])
                    obj = str(proposition[2])
                elif len(proposition) == 4:
                    verb = str(proposition[1])
                    obj = str(proposition[2])
                    context = str(proposition[3])

                triples.append({'subject': subject, 'verb': verb, 'object': obj, 'context': context})
    return triples