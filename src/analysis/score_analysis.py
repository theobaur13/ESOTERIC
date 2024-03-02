import os
import sqlite3
from evidence_retrieval.tools.document_retrieval import text_match_search, title_match_search, score_docs
from evidence_retrieval.tools.NER import extract_entities
from tqdm import tqdm
from transformers import pipeline
import sentence_transformers
import json
import spacy
import re

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
    return cursor, NER_pipe, conn

def disambig_scoring(database_path, output_dir, preloaded_claim=None):
    output_path = os.path.join(output_dir, "disambiguation_score_results.json")
    cursor, NER_pipe, conn = initialiser(database_path, preloaded_claim)
    nlp = spacy.load('en_core_web_sm')

    for row in tqdm(cursor.fetchall()):
        # if target doc does not contain the pattern "_-LRB-" + string + "-RRB-", then move to next claim
        target_doc = row[0]
        if not re.search(r'_-LRB-.+-RRB-', target_doc):
            continue

        claim = row[1]
        print("Scoring claim:", row[1])

        result = {
            "claim": claim,
            "target_doc": target_doc,
            "evidence": []
        }

        entities = extract_entities(NER_pipe, claim)
        for entity in entities:
            docs = title_match_search(entity, conn)
            docs = score_docs(docs, claim, nlp)
            docs = [doc for doc in docs if doc['score'] != 1]
            docs = sorted(docs, key=lambda x: x['score'], reverse=True)

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

        write_to_file(result, output_path)

def text_match_scoring(database_path, output_dir, preloaded_claim=None):
    output_path = os.path.join(output_dir, "text_matching_score_results.json")
    cursor, NER_pipe, conn = initialiser(database_path, preloaded_claim)
    encoder = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")

    limit = 1000
    k = 1000

    for row in tqdm(cursor.fetchall()):
        print("Scoring claim:", row[1])
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
        write_to_file(result, output_path)

def write_to_file(data, output_path):
    if not os.path.exists(output_path):
        with open(output_path, 'w') as file:
            json.dump([], file) 

    with open(output_path, 'r') as file:
        old_data = json.load(file)

    old_data.append(data)
    
    with open(output_path, 'w') as file:
        json.dump(old_data, file, indent=4)