import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
import sqlite3
import numpy as np
import polars as pl

def TF_IDF(claim):
    corpus = [claim]
    vectoriser = TfidfVectorizer(strip_accents='ascii')
    TF_IDF_matrix = vectoriser.fit_transform(corpus)
    feature_names = vectoriser.get_feature_names_out()
    df = pd.DataFrame(TF_IDF_matrix.toarray(), columns=feature_names)
    return df

def cosine_similarity(claim_TF_IDF_matrix, conn):
    # connect to the database
    cursor = conn.cursor()

    # get list of all unique relevant terms in the tf_idf table
    claim_terms = list(claim_TF_IDF_matrix.columns)
    formatted_terms = ','.join(f"'{term}'" for term in claim_terms)

    # get list of all unique relevant doc_ids in the tf_idf table
    cursor.execute(f"""
        SELECT DISTINCT doc_id
        FROM tf_idf
        WHERE term IN ({formatted_terms})
    """)
    doc_ids = [row[0] for row in cursor.fetchall()]
    print("Selected", len(doc_ids), "doc_ids")

    # get list of all unique relevant terms in the tf_idf table
    cursor.execute(f"""
        SELECT DISTINCT term
        FROM tf_idf
        WHERE doc_id IN (SELECT DISTINCT doc_id FROM tf_idf WHERE term IN ({formatted_terms}))
    """)
    terms = [row[0] for row in cursor.fetchall()]

    # add all terms to the claim_TF_IDF_matrix
    missing_terms = list(set(terms) - set(claim_TF_IDF_matrix.columns))
    if missing_terms:
        missing_terms_df = pd.DataFrame(0, index=claim_TF_IDF_matrix.index, columns=missing_terms)
        claim_TF_IDF_matrix = pd.concat([claim_TF_IDF_matrix, missing_terms_df], axis=1)

    # sort columns alphabetically
    claim_TF_IDF_matrix = claim_TF_IDF_matrix.reindex(sorted(claim_TF_IDF_matrix.columns), axis=1)
    
    # convert claim_TF_IDF_matrix to a polars series
    claim_TF_IDF_series = pl.Series(claim_TF_IDF_matrix.values[0])
    norm_claim = np.sqrt((claim_TF_IDF_series ** 2).sum())

    # split list of doc_ids into batches
    batch_size = 100
    doc_id_batches = [doc_ids[i:i+batch_size] for i in range(0, len(doc_ids), batch_size)]

    # 5 highest similarity scores and their corresponding doc_ids
    highest_similarity = []

    for batch in tqdm(doc_id_batches):
        cursor.execute("SELECT doc_id, term, tf_idf_score FROM tf_idf WHERE doc_id IN ({})".format(','.join(['?']*len(batch))), batch)
        data = cursor.fetchall()

        TF_IDF_data = pl.DataFrame(data, schema=['doc_id', 'term', 'tf_idf_score'])
        TF_IDF_data = TF_IDF_data.sort('term')
        TF_IDF_data = TF_IDF_data.pivot(index='doc_id', columns='term', values='tf_idf_score')

        for row in TF_IDF_data.iter_rows(named=True):
            doc_id = row['doc_id']
            row.pop('doc_id')  
            missing_terms = list(set(terms) - set(row.keys()))

            if missing_terms:
                for term in missing_terms:
                    row[term] = 0

            row = dict(sorted(row.items()))
            row_TF_IDF_series = pl.Series(row.values())

            # calculate cosine similarity
            dot = (row_TF_IDF_series * claim_TF_IDF_series).sum()
            norm_row = np.sqrt((row_TF_IDF_series ** 2).sum())
            similarity_score = dot / (norm_row * norm_claim)

            # add to highest_similarity if similarity_score is higher than the lowest score in highest_similarity
            if len(highest_similarity) < 5:
                highest_similarity.append((doc_id, similarity_score))
                highest_similarity = sorted(highest_similarity, key=lambda x: x[1], reverse=True)
            else:
                if similarity_score > highest_similarity[-1][1]:
                    highest_similarity.pop()
                    highest_similarity.append((doc_id, similarity_score))
                    highest_similarity = sorted(highest_similarity, key=lambda x: x[1], reverse=True)

    return highest_similarity