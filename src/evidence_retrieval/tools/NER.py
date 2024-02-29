def extract_entities(nlp, text):
    print("Extracting entities from text")
    ner_results = nlp(text)

    entities = []
    for entity in ner_results:
        entity_string = entity['word']

        if "-" in entity_string:
            entity_string = entity_string.replace(" - ", "-")
            
        entities.append(entity_string)

    print("Entities:", entities)
    return entities