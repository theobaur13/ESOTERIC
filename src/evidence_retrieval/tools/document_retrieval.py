import faiss
import sentence_transformers
import os

def match_search(query, conn):
    print("Searching for documents containing keyword '" + str(query) + "'")

    query = query.replace(' ', '_')

    cursor = conn.cursor()
    cursor.execute("SELECT id, doc_id FROM documents WHERE doc_id LIKE ?", ('%' + query + '%',))
    rows = cursor.fetchall()

    docs = []
    for row in rows:
        id = row[0]
        doc_id = row[1]
        docs.append({"id" : id, "doc_id" : doc_id})
    return docs

def FAISS_search(query, data_path):
    print("Searching for documents close to query '" + str(query) + "' using FAISS")
    model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L3-v2")
    query_vector = model.encode([query])
    k = 10
    index = faiss.read_index(os.path.join(data_path, 'faiss_index'))

    top_k = index.search(query_vector, k)
    ids = [int(i) for i in top_k[1][0]]
    search_results = dict(zip(ids, top_k[0][0]))
    
    return search_results