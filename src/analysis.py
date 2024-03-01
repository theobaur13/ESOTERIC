import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from analysis.retrieval_loader import retrieval_loader
from analysis.retrieval_analysis import skeleton

if __name__ == "__main__":
    load_or_analyze = input("Would you like to load the data or analyze it?\n(l) Load, (a) Analyze\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if load_or_analyze == "l":
        retrieval_loader()
    elif load_or_analyze == "a":
        database_path = os.path.join(current_dir, '..', 'data')
        output_dir = os.path.join(current_dir, '..', 'data', 'analysis')

        preloaded_claim = input("Enter a claim ID to analyze, or 'all' to analyze all claims:\n")
        # if preloaded claim is int
        if preloaded_claim.isdigit():
            preloaded_claim = int(preloaded_claim)
            skeleton(database_path, output_dir, preloaded_claim)
        else:
            skeleton(database_path, output_dir)
    else:
        print("Invalid input. Please enter 'l' or 'a'.")
