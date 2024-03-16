def extract_entities(NER_model, text):
    NER_results = NER_model.predict(text)
    entities = [entity['span'] for entity in NER_results]
    return entities