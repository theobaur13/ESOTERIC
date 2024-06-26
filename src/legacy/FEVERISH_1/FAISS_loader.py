import faiss
import sentence_transformers
import argparse
import os
import sqlite3
from tqdm import tqdm

def load_FAISS(batch_size=999):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    database_path = os.path.join(current_dir, 'data')
    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L6-v2")
    index = faiss.IndexIDMap(faiss.IndexFlatIP(384))

    # Load the documents from the database in batches
    cursor.execute("SELECT id FROM documents")
    ids = [row[0] for row in cursor.fetchall()]
    ids.sort()

    id_batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

    for batch in tqdm(id_batches):
        cursor.execute("SELECT text FROM documents WHERE id IN ({})".format(','.join('?' for _ in batch)), batch)
        data = cursor.fetchall()
        docs = [row[0] for row in data]

        encoded_docs = model.encode(docs)
        index.add_with_ids(encoded_docs, batch)
    
    faiss.write_index(index, os.path.join(database_path, 'faiss_index'))