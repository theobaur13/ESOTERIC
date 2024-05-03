import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import sqlite3
import numpy as np
import polars as pl
import time
import json

def load(file_ids):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', '..', '..', 'data', 'wiki-pages')
    database_path = os.path.join(current_dir, 'data')

    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS documents")
    cursor.execute("DROP TABLE IF EXISTS tf_idf")
    conn.commit()

    start_time = time.time()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        text TEXT NOT NULL);
        ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tf_idf(
    doc_id INTEGER NOT NULL,
    term TEXT NOT NULL,
    tf_idf_score REAL NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES documents(id),
    PRIMARY KEY(term, doc_id));
    ''')

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tf_idf_doc_id ON tf_idf(doc_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tf_idf_term_doc ON tf_idf(term, doc_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tf_idf_term ON tf_idf(term);")

    doc_ids = []
    documents = []

    TF_IDF_vectoriser = TfidfVectorizer(strip_accents='ascii')

    # import jsonl file by file into the db
    print("Loading " , file_ids, " documents into database")
    file_list = os.listdir(dataset_path)

    # only get files that are in the file_ids list (wiki-001.jsonl = 1, wiki-002.jsonl = 2, etc.)
    file_list = [file for file in file_list if int(file.split('-')[1].split('.')[0]) in file_ids]

    for file in tqdm(file_list):
        file_path = os.path.join(dataset_path, file)
        
        if file_path.endswith('.jsonl'):
            with open(file_path) as f:
                for line in f:
                    data = json.loads(line)
                    documents.append(data['text'])
                    doc_ids.append(data['id'])
                    cursor.execute("INSERT INTO documents (doc_id, text) VALUES (?, ?)", (data['id'], data['text']))
        else:
            raise ValueError('Unsupported file format')
    conn.commit()

    print("Initialising TF IDF matrix")
    TF_IDF_matrix = TF_IDF_vectoriser.fit_transform(documents)

    print("Applying feature names to TF IDF matrix")
    feature_names = TF_IDF_vectoriser.get_feature_names_out()

    print("Calculating TF-IDF values")
    bulk_data = []
    for doc_index, doc_id in enumerate(tqdm(doc_ids)):
        for term_index in TF_IDF_matrix[doc_index].nonzero()[1]:
            term = feature_names[term_index]
            tf_idf_score = TF_IDF_matrix[doc_index, term_index]
            bulk_data.append((doc_id, term, tf_idf_score))

    print("Loading TF-IDF values into database")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA cache_size = 1000000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA locking_mode = EXCLUSIVE")

    chunk_size = 100000
    total_chunks = len(bulk_data) // chunk_size + (1 if len(bulk_data) % chunk_size > 0 else 0)

    for i in tqdm(range(total_chunks)):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        chunk = bulk_data[start_index:end_index]
        cursor.executemany("INSERT INTO tf_idf (doc_id, term, tf_idf_score) VALUES ((SELECT id FROM documents WHERE doc_id = ?), ?, ?)", chunk)
    
    conn.commit()

    end_time = time.time()
    print("Database loaded successfully in " + str(end_time - start_time) + " seconds")