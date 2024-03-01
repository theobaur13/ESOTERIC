import faiss
import sentence_transformers
import os
import re
import faiss
from pandas import DataFrame as df

def title_match_search(query, conn):
    print("Searching for titles containing keyword '" + str(query) + "'")

    formatted_query = query.replace(' ', '_').lower()

    cursor = conn.cursor()
    cursor.execute("""
                   SELECT id, doc_id FROM documents WHERE LOWER(doc_id) = ?
                   UNION
                   SELECT id, doc_id FROM documents WHERE LOWER(doc_id) LIKE ?
                   """, (formatted_query, formatted_query + '_-LRB-%-RRB-%',))
    rows = cursor.fetchall()

    docs = []
    for row in rows:
        id = row[0]
        doc_id = row[1]
        docs.append({"id" : id, "doc_id" : doc_id})
    return docs

def text_match_search(claim, query, conn, encoder):
    print("Searching for documents containing keyword '" + str(query) + "'")

    formatted_query = query.lower()

    cursor = conn.cursor()
    cursor.execute("""
                SELECT id, doc_id, text FROM documents WHERE LOWER(text) LIKE ? LIMIT 100
                """, ("%" + formatted_query + "%",))
    rows = cursor.fetchall()

    if len(rows) == 0:
        return []

    data = df(rows, columns=['id', 'doc_id', 'text'])
    text = data['text'].tolist()
    
    doc_count = len(text)

    text_vectors = encoder.encode(text)
    claim_vector = encoder.encode([claim])

    index = faiss.IndexFlatIP(text_vectors.shape[1])
    index.add(text_vectors)

    k = min(10, doc_count)
    top_k = index.search(claim_vector, k)

    docs = []
    for i in range(k):
        doc_id = data['doc_id'][top_k[1][0][i]]
        score = top_k[0][0][i]
        id = int(data['id'][top_k[1][0][i]])
        docs.append({"id" : id, "doc_id" : doc_id, "score" : score, "method" : "text_match"})

    docs = sorted(docs, key=lambda x: x['score'], reverse=True)
    return docs
    

def score_docs(docs, query, nlp):
    print("Scoring documents")

    disambiguated_docs = []

    for doc in docs:
        doc_id = doc['doc_id']
        pattern = r'\-LRB\-.+\-RRB\-'

        if re.search(pattern, doc_id):
                disambiguated_docs.append(doc)
                docs = [d for d in docs if d['doc_id'] != doc_id]

    for doc in disambiguated_docs:
        doc_id = doc['doc_id']
    
        pattern = r'\-LRB\-(.+)\-RRB\-'
        info = re.search(pattern, doc_id).group(1)

        nlp_info = nlp(info)
        nlp_query = nlp(query)
        score = nlp_info.similarity(nlp_query)
        doc['score'] = score
        doc['method'] = "disambiguation"

    for doc in docs:
        doc['score'] = 1
        doc['method'] = "title_match"

    docs = docs + disambiguated_docs
    return docs