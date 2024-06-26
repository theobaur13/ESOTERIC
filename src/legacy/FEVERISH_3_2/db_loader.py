import os
import sqlite3
import json
import time
import argparse
from tqdm import tqdm

def main(batch_limit=None):
    # Set up the database connection
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', '..', '..', 'data', 'wiki-pages')
    database_path = os.path.join(current_dir, 'data')

    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    # Drop table if it exists
    cursor.execute("DROP TABLE IF EXISTS documents")
    conn.commit()

    start_time = time.time()

    # Create table [id] [doc_id] [text]
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        text TEXT NOT NULL);
        ''')

    # Create index on id and doc_id
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);")

    # Create virtual table for full text search
    cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING FTS5(doc_id, text);")

    doc_ids = []
    documents = []

    # import jsonl file by file into the db
    print("Loading " + str(batch_limit) + " documents into database")
    file_list = os.listdir(dataset_path)
    
    # Limit the number of files processed from command line arg
    if batch_limit:
        file_list = file_list[:batch_limit]
    
    # Load data into database file by file
    for file in tqdm(file_list):
        file_path = os.path.join(dataset_path, file)

        # Load data from jsonl files
        if file_path.endswith('.jsonl'):
            with open(file_path) as f:
                for line in f:
                    data = json.loads(line)
                    documents.append(data['text'])
                    doc_ids.append(data['id'])
                    cursor.execute("INSERT INTO documents (doc_id, text) VALUES (?, ?)", (data['id'], data['text']))
                    cursor.execute("INSERT INTO documents_fts (doc_id, text) VALUES (?, ?)", (data['id'], data['text']))
        else:
            raise ValueError('Unsupported file format')
    conn.commit()

    end_time = time.time()
    print("Database loaded successfully in " + str(end_time - start_time) + " seconds")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load wiki-pages into database.')
    parser.add_argument('--batch_limit', type=int, help='Limit the number of files processed. Leave blank to process all files.', nargs='?', const=None)
    args = parser.parse_args()
    
    main(batch_limit=args.batch_limit)