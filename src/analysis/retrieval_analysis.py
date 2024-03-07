import os
import sqlite3
import json
from tqdm import tqdm
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
import json

def initialiser(database_path, preloaded_claim=None):
    # Set up the database connection
    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    # Select claims to test from db
    if preloaded_claim:
        cursor.execute('''
            SELECT c.claim, GROUP_CONCAT(cd.doc_id) AS doc_ids
            FROM claims c
            JOIN claim_docs cd ON c.claim_id = cd.claim_id
            WHERE c.claim_id = ?
            GROUP BY c.claim_id
        ''', (preloaded_claim,))
    else:
        cursor.execute('''
            SELECT c.claim, GROUP_CONCAT(cd.doc_id) AS doc_ids
            FROM claims c
            JOIN claim_docs cd ON c.claim_id = cd.claim_id
            GROUP BY c.claim_id
            ORDER BY RANDOM()
        ''')

    # Initialise evidence retriever
    title_match_docs_limit = 1000
    text_match_search_db_limit = 1000
    text_match_search_k_limit = 100

    title_match_search_threshold = 0
    text_match_search_threshold = 0
    answerability_threshold = 0.1
    
    evidence_retriever = EvidenceRetriever(database_path, title_match_docs_limit=title_match_docs_limit, text_match_search_db_limit=text_match_search_db_limit, text_match_search_k_limit=text_match_search_k_limit, title_match_search_threshold=title_match_search_threshold, text_match_search_threshold=text_match_search_threshold, answerability_threshold=answerability_threshold)

    return cursor, evidence_retriever

def skeleton(database_path, output_dir, preloaded_claim=None):
    # Set output path for results
    output_path = os.path.join(output_dir, "retrieval_results.json")

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
        doc_ids = row[1].split(',')

        claim = row[0]
        hit_status = False

        print("\nTarget documents:", doc_ids)
        evidence_wrapper = evidence_retriever.retrieve_documents(claim)

        # Store the results in a dictionary
        result = {
            "claim": claim,
            "target_doc": doc_ids,
            "evidence": []
            }

        # Check if any documents in the evidence wrapper match the target document
        for evidence in evidence_wrapper.get_evidences():
            # Add evidence to the result dictionary
            result["evidence"].append({
                "entity" : evidence.entity,
                "doc_id": evidence.doc_id,
                "score": str(evidence.doc_score),
                "method" : evidence.doc_retrieval_method,
                "hit" : False
            })

            # If the evidence matches the target document, set the hit status to true
            if evidence.doc_id in doc_ids:
                hit_status = True
                for e in result["evidence"]:
                    if e["doc_id"] == evidence.doc_id:
                        e["hit"] = True
                print("Hit on document:", evidence.doc_id)
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
        if not os.path.exists(output_path):
            with open(output_path, 'w') as file:
                json.dump([], file) 

        with open(output_path, 'r') as file:
            data = json.load(file)

        data.append(result)
    
        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)
