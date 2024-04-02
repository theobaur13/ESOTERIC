import torch

# These functions are unused in the current implementation of the project
def extract_focals(nlp, text):
    print("Extracting focal points from text")
    doc = nlp(text)

    tag_set = {
        "all": ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"],
    }

    active_tag_set = tag_set["all"]
    focals = []

    for entity in doc.ents:
        if entity.label_ in active_tag_set:
            focals.append({'entity': entity.text, 'type': entity.label_})

    return focals

def extract_questions(nlp, focal_point, claim):
    question_generation_string = "answer: " + focal_point + " context: " + claim
    question_generation_output = nlp(question_generation_string)
    question = question_generation_output[0]['generated_text'].replace("question: ", "")
    return question

def calculate_answerability_score(tokeniser, model, context, question):
    input_string = question + " " + tokeniser.sep_token + " " + context
    encoded_input = tokeniser(input_string, return_tensors="pt", truncation=True)
    prob = torch.sigmoid(model(**encoded_input).logits.squeeze(-1)).item()
    return prob

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