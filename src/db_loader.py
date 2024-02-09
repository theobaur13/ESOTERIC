import os
import sqlite3
import json

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'data', 'wiki-pages')
    database_path = os.path.join(current_dir, '..', 'data')

    conn = sqlite3.connect(os.path.join(database_path, 'wiki-pages.db'))

    # import jsonl file by file into the db
    file_list = os.listdir(dataset_path)
    for file in file_list:
        file_path = os.path.join(dataset_path, file)
        print ("Loading file: " + file)
        
        if file_path.endswith('.jsonl'):
            with open(file_path) as f:
                for line in f:
                    data = json.loads(line)
                    conn.execute('''
                                CREATE TABLE IF NOT EXISTS documents(
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                doc_id TEXT NOT NULL,
                                text TEXT NOT NULL);
                                ''')
                    conn.execute("INSERT INTO documents (doc_id, text) VALUES (?, ?)", (data['id'], data['text']))
        else:
            raise ValueError('Unsupported file format')
    conn.commit()

if __name__ == '__main__':
    main()