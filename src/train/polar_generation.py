import os
import sqlite3
import json
from tqdm import tqdm
from evidence_retrieval.tools.NER import extract_entities
from span_marker import SpanMarkerModel

def create_dataset(database_path, output_dir):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    model = SpanMarkerModel.from_pretrained("lxyuan/span-marker-bert-base-multilingual-uncased-multinerd")

    # Select all claims alongside their target doc_ids, grouped by claim_id
    cursor.execute("""
                SELECT c.claim, GROUP_CONCAT(cd.doc_id, ';') AS doc_ids
                FROM claims c
                JOIN claim_docs cd ON c.claim_id = cd.claim_id
                GROUP BY c.claim_id
                   """)
    
    results = []

    for row in tqdm(cursor.fetchall()):
        claim = row[0]
        doc_ids = list(set(row[1].split(';')))
        entities = extract_entities(model, claim)

        results.append({"claim": claim, "doc_ids": doc_ids, "entities": entities})

    # Write the results to a json file
    output_path = os.path.join(output_dir, 'entities.json')

    with open(output_path, 'w') as file:
        json.dump(results, file)