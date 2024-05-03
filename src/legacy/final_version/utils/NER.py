def extract_entities(answer_pipe, NER_pipe, text):
    # Extract entities from text through pipeline
    input = "extract entities: <ha> " + text + " <ha>"
    output = answer_pipe(input)

    entities = []
    answers = output[0]['generated_text'].split("<sep>")

    for answer in answers:
        if answer != "":
            entities.append(answer.strip())

    # Extract entities from text through NER
    NER_results = NER_pipe(text)
    for entity in NER_results:
        entity_string = str(entity['word'])

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

    # Remove duplicates
    entities = list(set(entities))

    # Remove entities that contain hashtags and only contain numbers
    entities = [entity for entity in entities if "#" not in entity and not entity.isdigit()]
    return entities