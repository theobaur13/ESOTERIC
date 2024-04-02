import os
import sqlite3
import json
import time
import argparse
from tqdm import tqdm

def main(batch_limit=None):
    # Set up the database connection
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'data', 'wiki-pages')
    claims_path = os.path.join(current_dir, '..', 'data', 'claims')
    database_path = os.path.join(current_dir, '..', 'data')

    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    start_time = time.time()

    # Drop table if it exists
    print("Dropping existing tables")
    cursor.execute("DROP TABLE IF EXISTS documents")
    conn.commit()
    cursor.execute("DROP TABLE IF EXISTS claims")
    conn.commit()
    cursor.execute("DROP TABLE IF EXISTS claim_docs")
    conn.commit()
    print("Dropped existing tables")

    # Create table [id] [doc_id] [text]
    print("Creating new documents table")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        text TEXT NOT NULL,
        file_no INTEGER DEFAULT 0);
        ''')

    # Create index on id and doc_id
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);")
    print("Created new documents table")
    
    # Limit the number of files processed from command line arg
    file_list = os.listdir(dataset_path)
    if batch_limit:
        file_list = file_list[:batch_limit]

    # Load data into database file by file
    print("Loading " + str(len(file_list)) + " wiki-page files into database")
    for file in tqdm(file_list):
        file_path = os.path.join(dataset_path, file)
        file_no = int(file.split('-')[1].split('.')[0])

        # Load data from jsonl files
        if file_path.endswith('.jsonl'):
            with open(file_path) as f:
                for line in f:
                    data = json.loads(line)
                    cursor.execute("INSERT INTO documents (doc_id, text, file_no) VALUES (?, ?, ?)", (data['id'], data['lines'], file_no))
        else:
            raise ValueError('Unsupported file format')
    conn.commit()
    print("Finished loading wiki-pages into database")

    # Create table for claims
    print("Creating new claims table")
    cursor.execute("CREATE TABLE IF NOT EXISTS claims (id INTEGER PRIMARY KEY, claim_id INTEGER, claim TEXT NOT NULL);")

    # Create table for claim_docs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claim_docs (
            claim_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
            sent_id INTEGER NOT NULL,
            doc_no INTEGER DEFAULT 0,
            FOREIGN KEY (claim_id) REFERENCES claims (claim_id),
            FOREIGN KEY (doc_id) REFERENCES documents (doc_id),
            PRIMARY KEY (claim_id, doc_id, sent_id)
        );
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS claim_id_index ON claims (claim_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS claim_docs_claim_id_index ON claim_docs (claim_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS claim_docs_doc_id_index ON claim_docs (doc_id);")
    print("Created new claims table")

    print("Loading claims into database")
    file_list = os.listdir(claims_path)
    for file in file_list:
        print("Loading " + file + " into database")
        file_path = os.path.join(claims_path, file)

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
                                    # get doc_no from documents table
                                    cursor.execute("SELECT file_no FROM documents WHERE doc_id = ?", (doc[2],))
                                    results = cursor.fetchone()
                                    if results:
                                        doc_no = results[0]
                                    cursor.execute("INSERT INTO claim_docs (claim_id, doc_id, sent_id, doc_no) VALUES (?, ?, ?, ?)", (data["id"], doc[2], doc[3], doc_no))
                                except sqlite3.IntegrityError:
                                    pass
    conn.commit()
    print("Finished loading claims into database")

    end_time = time.time()
    print("Database loaded successfully in " + str(end_time - start_time) + " seconds")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load wiki-pages into database.')
    parser.add_argument('--batch_limit', type=int, help='Limit the number of files processed. Leave blank to process all files.', nargs='?', const=None)
    args = parser.parse_args()
    
    main(batch_limit=args.batch_limit)