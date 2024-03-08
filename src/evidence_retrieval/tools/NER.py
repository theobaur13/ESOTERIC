def extract_entities(nlp, text):
    print("Extracting entities from text")
    
    # Extract entities from text through pipeline
    ner_results = nlp(text)

    entities = []
    for entity in ner_results:
        entity_string = entity['word']

        # For entities with a hyphen, remove spaces around hyphen, e.g. "Spider - man" -> "Spider-man"
        if "-" in entity_string:
            entity_string = entity_string.replace(" - ", "-")
        elif " ' " in entity_string:
            entity_string = entity_string.replace(" ' ", "'")
        elif " : " in entity_string:
            entity_string = entity_string.replace(" : ", ": ")
        elif " , " in entity_string:
            entity_string = entity_string.replace(" , ", ",")
        elif " . " in entity_string:
            entity_string = entity_string.replace(" . ", ".")
            
        entities.append(entity_string)

    print("Entities:", entities)
    return entities