from TF_IDF import word_extraction, textual_tokenization, TF, IDF, create_matrix
import os
import sqlite3

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import dask.dataframe as dd 

def TF_IDF(corpus):
    corpus = word_extraction(corpus)
    corpus = textual_tokenization(corpus)
    tf = TF(corpus)
    idf = IDF(corpus)
    matrix = create_matrix(corpus, tf, idf)
    return matrix

# def TF_IDF(corpus):
#     vectorizer = CountVectorizer()
#     word_count = vectorizer.fit_transform(corpus)

#     tf = dd.from_array(word_count.toarray(), columns=vectorizer.get_feature_names_out())
    
#     transformer = TfidfTransformer()
#     X = transformer.fit_transform(word_count)
#     idf = dd.from_array(X.toarray(), columns=vectorizer.get_feature_names_out())
#     tf_idf = tf * idf
#     return tf_idf

def cosine_similarity(claim):
    pass
