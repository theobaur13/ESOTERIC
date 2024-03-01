import os
import sqlite3
from evidence_retrieval.tools.document_retrieval import text_match_search
from evidence_retrieval.tools.NER import extract_entities
from tqdm import tqdm
from transformers import pipeline
import sentence_transformers
import json

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

    NER_pipe = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)
    encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
    return cursor, NER_pipe, conn, encoder

def text_match_scoring(database_path, output_dir, preloaded_claim=None):
    output_path = os.path.join(output_dir, "text_matching_score_results.json")
    cursor, NER_pipe, conn, encoder = initialiser(database_path, preloaded_claim)

    limit = 1000
    k = 1000

    for row in tqdm(cursor.fetchall()):
        target_doc = row[0]
        claim = row[1]

        result = {
            "claim": claim,
            "target_doc": target_doc,
            "evidence": []
        }

        entities = extract_entities(NER_pipe, claim)
        for entity in entities:
            docs = text_match_search(claim, entity, conn, encoder, limit, k)
            for doc in docs:
                id = doc['id']
                doc_id = doc['doc_id']
                score = doc['score']
                entity = doc['entity']

                if doc_id == target_doc:
                    hit_status = True
                else:
                    hit_status = False

                result["evidence"].append({
                    "id": str(id),
                    "doc_id": doc_id,
                    "score": str(score),
                    "entity": entity,
                    "hit_status": hit_status
                })

    # Write results to file
    with open(output_path, 'r') as file:
        data = json.load(file)

    data.append(result)
    
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)


