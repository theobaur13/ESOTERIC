import os
import sqlite3
import spacy
import claucy
import json
from tqdm import tqdm
from query_generation.tools.varifocal import extract_focals, generate_questions
from query_generation.tools.NER import extract_entities
from query_generation.tools.triple_extraction import extract_triples
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
from models import Query, QueryWrapper
from transformers import pipeline

def initialiser(database_path):
    conn = sqlite3.connect(os.path.join(database_path, 'wiki-pages.db'))
    cursor = conn.cursor()

    # cursor.execute('''
    #     SELECT documents.id, test_retrieval.claim
    #     FROM test_retrieval
    #     JOIN documents ON test_retrieval.doc_id = documents.doc_id
    # ''')

    cursor.execute('''
        SELECT documents.id, test_retrieval.claim
        FROM test_retrieval
        JOIN documents ON test_retrieval.doc_id = documents.doc_id
        ORDER BY RANDOM()
    ''')

    evidence_retriever = EvidenceRetriever(database_path)

    return cursor, evidence_retriever

def get_NC_queries(claim):
    query_wrapper = QueryWrapper(claim)
    query = Query(claim)
    query_wrapper.add_query(query)
    return query_wrapper

def get_VF_queries(claim, nlp, pipe):
    focals = extract_focals(nlp, claim)
    questions = generate_questions(pipe, claim, focals)

    query_wrapper = QueryWrapper(claim)

    for question in questions:
        query = Query(question)
        query.set_creation_method("VF")
        query_wrapper.add_query(query)

    return query_wrapper

def get_NER_queries(claim, pipe):
    entities = extract_entities(pipe, claim)

    query_wrapper = QueryWrapper(claim)

    for entity in entities:
        query = Query(entity)
        query.set_creation_method("NER")
        query_wrapper.add_query(query)

    return query_wrapper

def get_TE_queries(claim, nlp):
    triples = extract_triples(nlp, claim)

    query_wrapper = QueryWrapper(claim)

    for triple in triples:
        query = Query(str(triple['subject']) + " " + str(triple['verb']) + " " + str(triple['object']) + " " + str(triple['context']))
        query.set_creation_method("TE")
        query_wrapper.add_query(query)

    return query_wrapper

def get_combined_queries(claim, nlp, varifocal_pipe, ner_pipe):
    focals = extract_focals(nlp, claim)
    questions = generate_questions(varifocal_pipe, claim, focals)
    entities = extract_entities(ner_pipe, claim)
    triples = extract_triples(nlp, claim)

    query_wrapper = QueryWrapper(claim)

    for question in questions:
        query = Query(question)
        query.set_creation_method("VF")
        query_wrapper.add_query(query)

    for entity in entities:
        query = Query(entity)
        query.set_creation_method("NER")
        query_wrapper.add_query(query)

    for triple in triples:
        query = Query(str(triple['subject']) + " " + str(triple['verb']) + " " + str(triple['object']) + " " + str(triple['context']))
        query.set_creation_method("TE")
        query_wrapper.add_query(query)

    return query_wrapper

def skeleton(database_path, method, output_dir):
    cursor, evidence_retriever = initialiser(database_path)
    if method == "VF":
        nlp = spacy.load("en_core_web_sm")
        pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
    elif method == "NER":
        pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
    elif method == "TE":
        nlp = spacy.load("en_core_web_sm")
        claucy.add_to_pipe(nlp)
    elif method == "NC":
        pass
    elif method == "C":
        nlp = spacy.load("en_core_web_sm")
        varifocal_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")
        ner_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
        claucy.add_to_pipe(nlp)
    else:
        print("Invalid method. Please enter 'VF' or 'NER' or 'TE' or 'NC' or 'C'.")
        return
    
    hits = 0
    misses = 0
    results = []

    for row in tqdm(cursor.fetchall()):
        doc_id = row[0]
        claim = row[1]

        if method == "VF":
            query_wrapper = get_VF_queries(claim, nlp, pipe)
        elif method == "NER":
            query_wrapper = get_NER_queries(claim, pipe)
        elif method == "TE":
            query_wrapper = get_TE_queries(claim, nlp)
        elif method == "NC":
            query_wrapper = get_NC_queries(claim)
        elif method == "C":
            query_wrapper = get_combined_queries(claim, nlp, varifocal_pipe, ner_pipe)

        hit_status = False

        for query in query_wrapper.get_subqueries():
            evidence_wrapper = evidence_retriever.retrieve_documents(query)

            for evidence in evidence_wrapper.get_evidences():
                if evidence.doc_id == doc_id:
                    hit_status = True
                    result = {
                        "method": method,
                        "claim": claim,
                        "query": query.text,
                        "target_doc_id": doc_id,
                        "actual_doc_id": evidence.doc_id,
                        "hit/miss": "hit"
                    }
                    print("Hit on document:", doc_id, "with query:", query.text, "using method:", method)
                    break
                else:
                    result = {
                        "method": method,
                        "claim": claim,
                        "query": query.text,
                        "target_doc_id": doc_id,
                        "actual_doc_id": evidence.doc_id,
                        "hit/miss": "miss"
                    }
            results.append(result)
                
        if hit_status:
            hits += 1
            print("Hit")
        else:
            misses += 1
            print("Miss")

        print(method + " hitrate: " + str((hits / (hits + misses)) * 100) + "%\n")

    output_path = os.path.join(output_dir, method + "_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)