import pandas as pd
import faiss
import sentence_transformers
import os

def triple_extraction(claim, nlp):
    print("Extracting triples from claim:", claim)
    triple = pd.DataFrame(columns=["subject", "predicate", "object"])
    doc = nlp(claim)

    sentences = doc.sents
    sentence_index = 0

    for sentence in sentences:
        sentence._.clauses
        propositions = sentence._.clauses[0].to_propositions(as_text=True)
        print(propositions)

        sentence_index += 1

def FAISS_search(claim, data_path):
    print("Searching for documents close to triple '" + str(claim) + "' using FAISS")
    model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L6-v2")
    query_vector = model.encode([claim])
    k = 5
    index = faiss.read_index(os.path.join(data_path, 'faiss_index'))

    top_k = index.search(query_vector, k)
    ids = [int(i) for i in top_k[1][0]]
    search_results = dict(zip(ids, top_k[0][0]))
    
    return search_results