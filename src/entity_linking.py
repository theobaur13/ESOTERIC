import pandas as pd
import faiss
import sentence_transformers
import os

def triple_extraction(claim, nlp):
    print("Extracting triples from claim")
    triple = pd.DataFrame(columns=["subject", "predicate", "object"])
    doc = nlp(claim)
    sentences = doc.sents

    sentence_index = 0
    for sentence in sentences:
        subject = ""
        predicate = "verb"
        object = ""

        for token in sentence:
            if token.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"):
                subject = token.text
            elif token.dep_ in ("ROOT", "conj", "xcomp", "ccomp", "advcl", "parataxis"):
                predicate = token.text
            elif token.dep_ in ("dobj", "dative", "attr", "oprd", "pobj"):
                object = token.text

        if subject and predicate and object:
            triple.loc[sentence_index] = [subject, predicate, object]

        sentence_index += 1

def FAISS_search(claim, data_path):
    print("Searching for documents close to triple using FAISS")
    model = sentence_transformers.SentenceTransformer("paraphrase-MiniLM-L6-v2")
    query_vector = model.encode([claim])
    k = 5
    index = faiss.read_index(os.path.join(data_path, 'faiss_index'))

    top_k = index.search(query_vector, k)
    ids = [int(i) for i in top_k[1][0]]
    return ids