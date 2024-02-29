import os
import sqlite3
import spacy
import claucy
import json
from tqdm import tqdm
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
from transformers import pipeline

def initialiser(database_path):
    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    cursor.execute('''
        SELECT documents.doc_id, test_retrieval.claim
        FROM test_retrieval
        JOIN documents ON test_retrieval.doc_id = documents.doc_id
        ORDER BY RANDOM()
    ''')

    evidence_retriever = EvidenceRetriever(database_path)

    return cursor, evidence_retriever

def skeleton(database_path, output_dir):
    cursor, evidence_retriever = initialiser(database_path)
    
    hits = 0
    misses = 0
    results = []

    for row in tqdm(cursor.fetchall()):
        doc_id = row[0]
        claim = row[1]

        hit_status = False

        evidence_wrapper = evidence_retriever.retrieve_documents(claim)

        for evidence in evidence_wrapper.get_evidences():
            if evidence.doc_id == doc_id:
                hit_status = True
                result = {
                    "claim": claim,
                    "target_doc_id": doc_id,
                    "actual_doc_id": evidence.doc_id,
                    "hit/miss": "hit"
                }
                print("Hit on document:", doc_id)
                break
            else:
                result = {
                    "claim": claim,
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

        print("Hitrate: " + str((hits / (hits + misses)) * 100) + "%\n")

    output_path = os.path.join(output_dir, + "results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)