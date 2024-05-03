import faiss
import re
import torch
from pandas import DataFrame as df

# Retrieve documents with exact title match inc. docs with disambiguation in title
def title_match_search(queries, es):
    print("Searching for titles containing keywords:", queries)

    # Convert query to lowercase and replace spaces with underscores
    formatted_queries = [query.replace(' ', '_').replace(':', '-COLON-').lower() for query in queries] 

    should_conditions = []
    for query in formatted_queries:
        should_conditions.append({
            "term": {
                "doc_id": query
            }
        })
        should_conditions.append({
            "wildcard": {
                "doc_id": {
                    "value": f"{query}_-LRB-*"
                }
            }
        })

    query_body = {
        "query": {
            "bool": {
                "should": should_conditions,
                "minimum_should_match": 1
            }
        }
    }

    response = es.search(index="documents", body=query_body)

    docs = []
    for hit in response['hits']['hits']:
        id = hit['_id']
        doc_id = hit['_source']['doc_id']
        text = hit['_source']['content']
        embedding = hit['_source']['embedding']
        docs.append({"id" : id, "doc_id" : doc_id, "entity" : query, "text" : text, "embedding" : embedding})
    return docs


def text_match_search(entities, es, limit=100):
    print("Searching for documents containing keywords:", entities)

    # Retrieve documents from db containing query
    query_body = {
        "query": {
            "bool": {
                "should": [
                    {"match_phrase": {"content": entity}}
                    for entity in entities
                ],
                "minimum_should_match": 1
            }
        },
        "size": limit
    }

    response = es.search(index="documents", body=query_body)
    rows = response['hits']['hits']

    # Return empty list if no documents found
    if len(rows) == 0:
        return []
    
    # Add to docs
    docs = []
    for hit in rows:
        id = hit['_id']
        doc_id = hit['_source']['doc_id']
        text = hit['_source']['content']
        embedding = hit['_source']['embedding']
        docs.append({"id" : id, "doc_id" : doc_id, "entity" : entities, "text" : text, "embedding" : embedding, "score": 0, "method": "text_match"})
    return docs

# Score title matched and disambiguated docs
def score_docs(docs, query, nlp):
    print("Scoring documents")

    # Disambiguate documents with disambiguation in title e.g. "Frederick Trump (businessman)"
    disambiguated_docs = []
    for doc in docs:
        doc_id = doc['doc_id']
        pattern = r'\-LRB\-.+\-RRB\-'

        if re.search(pattern, doc_id):
                disambiguated_docs.append(doc)
                docs = [d for d in docs if d['doc_id'] != doc_id]

    # Score disambiguated docs by cosine similarity between disambiguated info and query
    for doc in disambiguated_docs:
        doc_id = doc['doc_id']
    
        pattern = r'\-LRB\-(.+)\-RRB\-'
        info = re.search(pattern, doc_id).group(1)
        info = info.replace('_', ' ')

        nlp_info = nlp(info)
        nlp_query = nlp(query)
        score = nlp_info.similarity(nlp_query)
        doc['score'] = score
        doc['method'] = "disambiguation"

    # Score exact match docs with 1
    for doc in docs:
        doc['score'] = 1
        doc['method'] = "title_match"

    # Combine exact match and disambiguated docs
    docs = docs + disambiguated_docs
    return docs

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
            focals.append({'focal': entity.text, 'type': entity.label_})

    return focals

def extract_answers(pipe, context):
    input = "extract answers: <ha> " + context + " <ha>"
    output = pipe(input)

    focals = []
    answers = output[0]['generated_text'].split("<sep>")
    for answer in answers:
        if answer != "":
            focals.append({'focal': answer, 'type': "ANSWER"})
    return focals

def extract_questions(nlp, focal_point, claim):
    question_generation_string = "answer: " + focal_point + " context: " + claim
    question_generation_output = nlp(question_generation_string)
    question = question_generation_output[0]['generated_text'].replace("question: ", "")
    return question

def extract_polar_questions(nlp, pipe, claim):
    doc = nlp(claim)
    questions = []

    for sentence in doc.sents:
        altered = False
        for token in sentence:
            if token.dep_ == "ROOT":
                if token.pos_ == "AUX":
                    # remove the auxiliary verb and add to the beginning of the sentence
                    question = sentence.text.replace(token.text, "")
                    question = token.text + " " + question
                    altered = True
                elif token.pos_ == "VERB":
                    # if there is an auxiliary verb, remove it and add to the beginning of the sentence
                    aux = [child for child in token.children if child.dep_ == "aux"]
                    if aux:
                        question = sentence.text.replace(aux[0].text, "")
                        question = aux[0].text + " " + question
                        altered = True
        if not altered:
            input_string = "answer: " + "No" + " context: " + claim
            output = pipe(input_string)
            question = output[0]['generated_text'].replace("question: ", "")

        # capitalize the first letter of the question
        question = question[0].upper() + question[1:]

        # remove the period at the end of the question if it exists and add a question mark
        if question[-1] == ".":
            question = question[:-1] + "?"

        # remove any double spaces
        question = question.replace("  ", " ")
        
        questions.append(question)

    return questions

def calculate_answerability_score_SelfCheckGPT(tokeniser, model, context, question):
    input_string = question + " " + tokeniser.sep_token + " " + context
    encoded_input = tokeniser(input_string, return_tensors="pt", truncation=True)
    prob = torch.sigmoid(model(**encoded_input).logits.squeeze(-1)).item()
    return prob

def calculate_answerability_score_tiny(nlp, context, question):
    input = {
        'question': question,
        'context': context
    }
    output = nlp(input)
    score = output['score']
    return score