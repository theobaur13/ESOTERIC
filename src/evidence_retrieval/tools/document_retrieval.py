import faiss
import sentence_transformers
import os
import re
import faiss
from pandas import DataFrame as df

# Retrieve documents with exact title match inc. docs with disambiguation in title
def title_match_search(query, conn):
    print("Searching for titles containing keyword '" + str(query) + "'")

    # Convert query to lowercase and replace spaces with underscores
    formatted_query = query.replace(' ', '_').lower()

    # Retrieve documents from db
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT id, doc_id FROM documents WHERE LOWER(doc_id) = ?
                   UNION
                   SELECT id, doc_id FROM documents WHERE LOWER(doc_id) LIKE ?
                   """, (formatted_query, formatted_query + '_-LRB-%-RRB-%',))
    rows = cursor.fetchall()

    # Return dictionary of {"id" : id, "doc_id" : doc_id}
    docs = []
    for row in rows:
        id = row[0]
        doc_id = row[1]
        docs.append({"id" : id, "doc_id" : doc_id, "entity" : query})
    return docs

def text_match_search(claim, query, conn, encoder, limit=100, k_lim=10):
    print("Searching for documents containing keyword '" + str(query) + "'")

    # Convert query to lowercase
    formatted_query = query.lower()

    # Retrieve documents from db containing query
    cursor = conn.cursor()
    cursor.execute("""
                SELECT id, doc_id, text FROM documents WHERE LOWER(text) LIKE ? LIMIT ?
                """, ("%" + formatted_query + "%", limit))
    rows = cursor.fetchall()
    if len(rows) == 0:
        return []

    # Convert rows to dataframe [id][doc_id][text]
    data = df(rows, columns=['id', 'doc_id', 'text'])

    text = data['text'].tolist()
    doc_count = len(text)

    # Encode document texts and claim
    text_vectors = encoder.encode(text)
    claim_vector = encoder.encode([claim])

    # Create FAISS dot product index and add document vectors
    index = faiss.IndexFlatIP(text_vectors.shape[1])
    index.add(text_vectors)

    # Search for top 10 documents with highest similarity to claim
    k = min(k_lim, doc_count)
    top_k = index.search(claim_vector, k)

    # Return dictionary of {"id" : id, "doc_id" : doc_id, "score" : score, "method" : "text_match"}
    docs = []
    for i in range(k):
        doc_id = data['doc_id'][top_k[1][0][i]]
        score = top_k[0][0][i]
        id = int(data['id'][top_k[1][0][i]])
        docs.append({"id" : id, "doc_id" : doc_id, "score" : score, "method" : "text_match", "entity" : query})

    # Return sorted list of documents by score
    docs = sorted(docs, key=lambda x: x['score'], reverse=True)
    return docs

def text_match_search_TF_IDF(claim, query, conn):
    pass

# Score title matched and disambiguated docs
def score_docs(docs, query, nlp):
    print("Scoring documents")

    # Disambiguate documents with disambiguation in title e.g. "Frederick Trump (businessman)"
    disambiguated_docs = []
    for doc in docs:
        doc_id = doc['doc_id']
        pattern = r'\-LRB\-.+\-RRB\-'

        if re.search(pattern, doc_id):
                disambiguated_docs.append(doc)
                docs = [d for d in docs if d['doc_id'] != doc_id]

    # Score disambiguated docs by cosine similarity between disambiguated info and query
    for doc in disambiguated_docs:
        doc_id = doc['doc_id']
    
        pattern = r'\-LRB\-(.+)\-RRB\-'
        info = re.search(pattern, doc_id).group(1)

        nlp_info = nlp(info)
        nlp_query = nlp(query)
        score = nlp_info.similarity(nlp_query)
        doc['score'] = score
        doc['method'] = "disambiguation"

    # Score exact match docs with 1
    for doc in docs:
        doc['score'] = 1
        doc['method'] = "title_match"

    # Combine exact match and disambiguated docs
    docs = docs + disambiguated_docs
    return docs