import os
import sqlite3
import json
from tqdm import tqdm

def retrieval_loader():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', '..', 'data', 'claims')
    database_path = os.path.join(current_dir, '..', '..', 'data')

    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS test_retrieval")
    conn.commit()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_retrieval(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        claim TEXT NOT NULL);
        ''')

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_retrieval_doc_id ON test_retrieval(doc_id);")

    file_list = os.listdir(dataset_path)

    for file in file_list:
        print("Loading " + file + " into database")
        file_path = os.path.join(dataset_path, file)

        if file_path.endswith('.jsonl'):
            with open(file_path) as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    if data["label"] == "SUPPORTS" or data["label"] == "REFUTES":
                        for evidence_set in data["evidence"]:
                            for evidence in evidence_set:
                                cursor.execute("INSERT INTO test_retrieval (doc_id, claim) VALUES (?, ?)", (evidence[2], data["claim"]))
    
    conn.commit()