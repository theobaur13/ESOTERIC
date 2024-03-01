import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from analysis.retrieval_loader import retrieval_loader
from analysis.retrieval_analysis import skeleton
from analysis.score_analysis import text_match_scoring

if __name__ == "__main__":
    load_or_analyze = input("Would you like to load the data or analyze it?\n(l) Load, (a) Analyze\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if load_or_analyze == "l":
        retrieval_loader()
    elif load_or_analyze == "a":
        database_path = os.path.join(current_dir, '..', 'data')
        output_dir = os.path.join(current_dir, '..', 'data', 'analysis')

        score_or_retrieval = input("Would you like to analyze retrieval performance or scoring?\n(r) Retrieval, (s) Scoring\n")

        preloaded_claim = input("Enter a claim ID to analyze, or 'all' to analyze all claims:\n")

        if score_or_retrieval == "r":
            if preloaded_claim.isdigit():
                preloaded_claim = int(preloaded_claim)
                skeleton(database_path, output_dir, preloaded_claim)
            else:
                skeleton(database_path, output_dir)
        elif score_or_retrieval == "s":
            if preloaded_claim.isdigit():
                preloaded_claim = int(preloaded_claim)
                text_match_scoring(database_path, output_dir, preloaded_claim)
            else:
                text_match_scoring(database_path, output_dir)
    else:
        print("Invalid input. Please enter 'l' or 'a'.")
