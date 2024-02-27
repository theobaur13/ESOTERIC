import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from analysis.retrieval_loader import retrieval_loader
from analysis.retrieval_analysis import varifocal_hitrate, NER_hitrate, triple_extraction_hitrate

if __name__ == "__main__":
    load_or_analyze = input("Would you like to load the data or analyze it? (l/a) ")
    if load_or_analyze == "l":
        retrieval_loader()
    elif load_or_analyze == "a":
        current_dir = os.path.dirname(os.path.abspath(__file__))
        database_path = os.path.join(current_dir, '..', 'data')
        method = input("Which method would you like to analyze? (VF/NER/TE) ")
        if method == "VF":
            varifocal_hitrate(database_path)
        elif method == "NER":
            NER_hitrate(database_path)
        elif method == "TE":
            triple_extraction_hitrate(database_path)
        else:
            print("Invalid input. Please enter 'VF', 'NER', or 'TE'.")
    else:
        print("Invalid input. Please enter 'l' or 'a'.")
