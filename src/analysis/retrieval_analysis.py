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
    output_path = os.path.join(output_dir, "results.jsonl")

    cursor, evidence_retriever = initialiser(database_path)
    
    hits = 0
    misses = 0

    for row in tqdm(cursor.fetchall()):
        doc_id = row[0]
        claim = row[1]
        hit_status = False

        print("\nTarget document:", doc_id)
        evidence_wrapper = evidence_retriever.retrieve_documents(claim)

        result = {
            "claim": claim,
            "target_doc": doc_id,
            "evidence": []
            }

        for evidence in evidence_wrapper.get_evidences():
            result["evidence"].append({
                "doc_id": evidence.doc_id,
                "score": str(evidence.score),
                "method" : evidence.doc_retrieval_method,
                "hit" : False
            })
            if evidence.doc_id == doc_id:
                hit_status = True
                for e in result["evidence"]:
                    if e["doc_id"] == evidence.doc_id:
                        e["hit"] = True
                print("Hit on document:", doc_id)
                break
                
        if hit_status:
            hits += 1
            print("Hit")
        else:
            misses += 1
            print("Miss")

        print("Hitrate: " + str((hits / (hits + misses)) * 100) + "%\n")

        #write line to file
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + "\n")
