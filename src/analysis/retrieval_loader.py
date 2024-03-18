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
    cursor.execute("DROP TABLE IF EXISTS claims")
    conn.commit()
    cursor.execute("DROP TABLE IF EXISTS claim_docs")
    conn.commit()

    # Create table for claims
    cursor.execute("CREATE TABLE IF NOT EXISTS claims (id INTEGER PRIMARY KEY, claim_id INTEGER, claim TEXT NOT NULL);")

    # Create table for claim_docs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claim_docs (
            claim_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            sent_id INTEGER NOT NULL,
            FOREIGN KEY (claim_id) REFERENCES claims (claim_id),
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id),
            PRIMARY KEY (claim_id, doc_id, sent_id)
        );
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS claim_id_index ON claims (claim_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS claim_id_index ON claim_docs (claim_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS doc_id_index ON claim_docs (doc_id);")

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
                        cursor.execute("INSERT INTO claims (claim_id, claim) VALUES (?, ?)", (data["id"], data["claim"]))

                        for set in data["evidence"]:
                            for doc in set:
                                try:
                                    cursor.execute("INSERT INTO claim_docs (claim_id, doc_id, sent_id) VALUES (?, ?, ?)", (data["id"], doc[2], doc[3]))
                                except sqlite3.IntegrityError:
                                    pass
    conn.commit()