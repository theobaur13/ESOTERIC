import os
import sqlite3
import json
from tqdm import tqdm

def retrieval_loader():
    # Set up the database connection
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', '..', 'data', 'claims')
    database_path = os.path.join(current_dir, '..', '..', 'data')

    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    # Drop table if it exists
    cursor.execute("DROP TABLE IF EXISTS test_retrieval")
    conn.commit()

    # Create table [id] [doc_id] [claim]
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_retrieval(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        claim TEXT NOT NULL,
        UNIQUE(doc_id, claim));
        ''')

    # Create index on doc_id
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_retrieval_doc_id ON test_retrieval(doc_id);")

    # Load data into database
    file_list = os.listdir(dataset_path)
    for file in file_list:
        print("Loading " + file + " into database")
        file_path = os.path.join(dataset_path, file)

        # Load data from jsonl files
        if file_path.endswith('.jsonl'):
            with open(file_path) as f:
                for line in tqdm(f):
                    data = json.loads(line)
                    if data["label"] == "SUPPORTS" or data["label"] == "REFUTES":
                        for evidence_set in data["evidence"]:
                            for evidence in evidence_set:
                                # Only insert unique doc_id, claim pairs
                                try:
                                    cursor.execute("INSERT INTO test_retrieval (doc_id, claim) VALUES (?, ?)", (evidence[2], data["claim"]))
                                except sqlite3.IntegrityError:
                                    pass
    
    conn.commit()