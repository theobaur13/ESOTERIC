import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from evidence_retrieval.evidence_retrieval import EvidenceRetriever

def main():
    # Set up the database connection
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data')

    input_claim = input("Enter claim: ")

    # Specify doc limit parameters
    title_match_docs_limit = 1000
    text_match_search_db_limit = 1000

    # Specify score threshold parameters
    title_match_search_threshold = 0
    answerability_threshold = 0.65

    # Retrieve evidence
    evidence_retriever = EvidenceRetriever(data_path, title_match_docs_limit=title_match_docs_limit, text_match_search_db_limit=text_match_search_db_limit, title_match_search_threshold=title_match_search_threshold, answerability_threshold=answerability_threshold)
    evidence_collection = evidence_retriever.retrieve_evidence(input_claim)
    evidence_collection.sort_by_doc_score()

    # Print evidence
    print("\n\033[1mBase claim: {}\033[0m".format(evidence_collection.get_claim()))
    for evidence in evidence_collection.get_evidences():
        print("\nDoc ID:", evidence.doc_id)
        # print("Evidence Document:\n", evidence.evidence_text)
        print("Document Score:", str(evidence.doc_score))
        for sentence in evidence.sentences:
            print("ID:", sentence.sent_id)
            print("Sentence:", sentence.sentence)
            print("Sentence Score:", sentence.score)

if __name__ == '__main__':
    main()