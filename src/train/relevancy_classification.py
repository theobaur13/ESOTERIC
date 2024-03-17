import sqlite3
from tqdm import tqdm

def create_dataset(database_path, output_dir, limit=10):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Select all claims alongside their the target document texts as well as sentence ids
    cursor.execute("""
                SELECT c.claim, d.text, cd.sent_id
                FROM claims c
                JOIN claim_docs cd ON c.claim_id = cd.claim_id
                JOIN documents d ON cd.doc_id = d.doc_id
                ORDER BY RANDOM()
                LIMIT ?
                   """, (limit,))
    
    dataset = {"claims": []}

    for row in cursor.fetchall():
        claim = row[0]
        text = row[1]
        sent_id = row[2]