import faiss
import sentence_transformers
import argparse
import os
import sqlite3
from tqdm import tqdm

def main(batch_size=999):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    database_path = os.path.join(current_dir, '..', 'data')
    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
    index = faiss.IndexIDMap(faiss.IndexFlatIP(384))

    # Load the documents from the database in batches
    cursor.execute("SELECT id FROM documents")
    ids = [row[0] for row in cursor.fetchall()]
    ids.sort()

    id_batches = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

    for batch in tqdm(id_batches):
        cursor.execute("SELECT doc_id FROM documents WHERE id IN ({})".format(','.join(['?']*len(batch))), batch)
        data = cursor.fetchall()

        doc_titles = [row[0] for row in data]

        encoded_doc_titles = model.encode(doc_titles)
        index.add_with_ids(encoded_doc_titles, batch)
    
    faiss.write_index(index, os.path.join(database_path, 'faiss_index'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load FAISS vectors of documents.')
    parser.add_argument('--batch_size', type=int, help='Limit the number of files processed at once. Leave blank to process all files.', nargs='?', const=None)
    args = parser.parse_args()
    main(batch_size=args.batch_size)