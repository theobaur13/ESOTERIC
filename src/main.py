import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
import sqlite3

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')
    conn = sqlite3.connect(os.path.join(data_path, 'data.db'))

    input_claim = input("Enter claim: ")

    evidence_retriever = EvidenceRetriever(data_path)
    evidence_collection = evidence_retriever.retrieve_evidence(input_claim)

    print("\n\033[1mBase claim: {}\033[0m".format(evidence_collection.get_claim()))
    for evidence in evidence_collection.get_evidences():
        evidence.set_wiki_url(conn)
        print("\nEvidence Sentence:", evidence.evidence_sentence)
        print("Evidence Document:", evidence.evidence_text)
        print("Score:", str(evidence.score))
        print("Doc ID:", evidence.doc_id)
        print("Wiki URL:", evidence.wiki_url)

if __name__ == '__main__':
    main()