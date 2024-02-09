from TF_IDF import word_extraction, textual_tokenization, TF, IDF, create_matrix
import os
import sqlite3

def TF_IDF(corpus):
    corpus = word_extraction(corpus)
    corpus = textual_tokenization(corpus)
    tf = TF(corpus)
    idf = IDF(corpus)
    matrix = create_matrix(corpus, tf, idf)
    return matrix

def cosine_similarity(claim):
    pass
