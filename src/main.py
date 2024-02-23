import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from claim_generation.query_generation import QueryGenerator
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
import sqlite3

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')

    input_claim = input("Enter claim: ")

    query_generator = QueryGenerator(input_claim)
    evidence_retriever = EvidenceRetriever(data_path)

    evidence_collection = []
    queries = query_generator.generate_queries()
    for query in queries.get_all():
        evidence = evidence_retriever.retrieve_evidence(query)
        evidence_collection.append(evidence)

    print("\n\033[1mBase claim: {}\033[0m".format(queries.get_base_claim().text))
    for query in queries.get_subqueries():
        print("Extracted Query:", query.text)
    for evidences in evidence_collection:
        print("\n\033[1mQuery: {}\033[0m".format(evidences.query.text))
        for evidence in evidences.get_evidences():
            evidence.set_wiki_url(sqlite3.connect(os.path.join(data_path, 'wiki-pages.db')))
            print("\nEvidence Sentence:", evidence.evidence_sentence)
            print("Evidence Document:", evidence.evidence_text)
            print("Score:", str(evidence.score))
            print("Doc ID:", evidence.doc_id)
            print("Wiki URL:", evidence.wiki_url)

if __name__ == '__main__':
    main()