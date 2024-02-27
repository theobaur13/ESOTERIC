import os
import sqlite3
import spacy
from tqdm import tqdm
from query_generation.tools.varifocal import extract_focals, generate_questions
from query_generation.tools.NER import extract_entities
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
from models import Query, QueryWrapper
from transformers import pipeline

def varifocal_hitrate(database_path):
    conn = sqlite3.connect(os.path.join(database_path, 'wiki-pages.db'))
    cursor = conn.cursor()

    nlp = spacy.load("en_core_web_sm")
    pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap")

    hits = 0
    misses = 0

    cursor.execute('''
        SELECT documents.id, test_retrieval.claim
        FROM test_retrieval
        JOIN documents ON test_retrieval.doc_id = documents.doc_id
    ''')

    evidence_retriever = EvidenceRetriever(database_path)

    for row in tqdm(cursor.fetchall()):
        doc_id = row[0]
        claim = row[1]

        focals = extract_focals(nlp, claim)
        questions = generate_questions(pipe, claim, focals)

        query_wrapper = QueryWrapper(claim)

        for question in questions:
            query = Query(question)
            query.set_creation_method("VF")
            query_wrapper.add_query(query)

        for query in query_wrapper.get_subqueries():
            evidence_wrapper = evidence_retriever.retrieve_documents(query)

            hit_status = False
            for evidence in evidence_wrapper.get_evidences():
                if evidence.doc_id == doc_id:
                    hit_status = True
                    break
                
            if hit_status:
                hits += 1
                print("Hit")
            else:
                misses += 1
                print("Miss")

        print("Varifocal hitrate: " + str(hits / (hits + misses)))

def NER_hitrate(database_path):
    conn = sqlite3.connect(os.path.join(database_path, 'wiki-pages.db'))
    cursor = conn.cursor()

    pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)

    hits = 0
    misses = 0

    cursor.execute('''
        SELECT documents.id, test_retrieval.claim
        FROM test_retrieval
        JOIN documents ON test_retrieval.doc_id = documents.doc_id
    ''')

    evidence_retriever = EvidenceRetriever(database_path)

    for row in tqdm(cursor.fetchall()):
        doc_id = row[0]
        claim = row[1]

        entities = extract_entities(pipe, claim)

        query_wrapper = QueryWrapper(claim)

        for entity in entities:
            query = Query(entity)
            query.set_creation_method("NER")
            query_wrapper.add_query(query)

        for query in query_wrapper.get_subqueries():
            evidence_wrapper = evidence_retriever.retrieve_documents(query)

            hit_status = False
            for evidence in evidence_wrapper.get_evidences():
                if evidence.doc_id == doc_id:
                    hit_status = True
                    break
                
            if hit_status:
                hits += 1
                print("Hit")
            else:
                misses += 1
                print("Miss")

        print("NER hitrate: " + str(hits / (hits + misses)))

def triple_extraction_hitrate(database_path):
    pass