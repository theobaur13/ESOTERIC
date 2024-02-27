import textacy
import claucy

def extract_triples(nlp, text):
    print("Extracting triples from text:", text)

    claucy.add_to_pipe(nlp)

    doc = nlp(text)
    sentences = doc.sents

    triples = []

    for sentence in sentences:
        sentence._.clauses
        for clause in sentence._.clauses:
            propositions = clause.to_propositions(as_text=False)
            for proposition in propositions:
                subject = proposition[0].text
                verb = proposition[1].text
                obj = proposition[2].text
                context = ""
                if len(proposition) == 4:
                    context = proposition[3]

                triples.append({'subject': subject, 'verb': verb, 'object': obj, 'context': context})
    return triples