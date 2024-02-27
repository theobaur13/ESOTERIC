import os
import sqlite3
import spacy
from tqdm import tqdm
from query_generation.tools.varifocal import extract_focals, generate_questions
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
        SELECT doc_id, claim
        FROM test_retrieval
        WHERE doc_id IN (SELECT doc_id FROM documents)
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
    pass

def triple_extraction_hitrate(database_path):
    pass