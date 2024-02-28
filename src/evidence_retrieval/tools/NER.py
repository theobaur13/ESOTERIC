def extract_entities(nlp, text):
    print("Extracting entities from text")
    ner_results = nlp(text)

    entities = []
    for entity in ner_results:
        entities.append(entity['word'])

    print("Entities:", entities)
    return entities