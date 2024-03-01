import os
import sqlite3
import json
from tqdm import tqdm
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
from transformers import pipeline

def initialiser(database_path, preloaded_claim=None):
    # Set up the database connection
    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    # Select claims to test from db
    if preloaded_claim:
        cursor.execute('''
            SELECT documents.doc_id, test_retrieval.claim
            FROM test_retrieval
            JOIN documents ON test_retrieval.doc_id = documents.doc_id
            WHERE test_retrieval.id = ?
        ''', (preloaded_claim,))
    else:
        cursor.execute('''
            SELECT documents.doc_id, test_retrieval.claim
            FROM test_retrieval
            JOIN documents ON test_retrieval.doc_id = documents.doc_id
            ORDER BY RANDOM()
        ''')

    # Initialise evidence retriever
    evidence_retriever = EvidenceRetriever(database_path)

    return cursor, evidence_retriever

def skeleton(database_path, output_dir, preloaded_claim=None):
    # Set output path for results
    output_path = os.path.join(output_dir, "results.jsonl")

    # Set up the database connection and evidence retriever
    if preloaded_claim:
        cursor, evidence_retriever = initialiser(database_path, preloaded_claim)
    else:
        cursor, evidence_retriever = initialiser(database_path)
    
    # Initialise hit and miss counters for hitrate calculation
    hits = 0
    misses = 0

    # Iterate through each claim in the test retrieval table randomly
    for row in tqdm(cursor.fetchall()):
        doc_id = row[0]
        claim = row[1]
        hit_status = False

        print("\nTarget document:", doc_id)
        evidence_wrapper = evidence_retriever.retrieve_documents(claim)

        # Store the results in a dictionary
        result = {
            "claim": claim,
            "target_doc": doc_id,
            "evidence": []
            }

        # Check if any documents in the evidence wrapper match the target document
        for evidence in evidence_wrapper.get_evidences():
            # Add evidence to the result dictionary
            result["evidence"].append({
                "entity" : evidence.entity,
                "doc_id": evidence.doc_id,
                "score": str(evidence.score),
                "method" : evidence.doc_retrieval_method,
                "hit" : False
            })

            # If the evidence matches the target document, set the hit status to true
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

        # Print hitrate after each claim
        print("Hitrate: " + str((hits / (hits + misses)) * 100) + "%\n")

        #write line to file
        with open(output_path, 'a') as f:
            f.write(json.dumps(result) + "\n")
