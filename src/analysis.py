import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from analysis.retrieval_loader import retrieval_loader
from analysis.retrieval_analysis import skeleton
from analysis.retrieval_plotter import relationship_plotter

if __name__ == "__main__":
    load_or_analyze = input("Would you like to load the data or analyze it?\n(l) Load, (a) Analyze, (p) Plot\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if load_or_analyze == "l":
        retrieval_loader()
    elif load_or_analyze == "a":
        database_path = os.path.join(current_dir, '..', 'data')
        method = input("Which method would you like to analyze? (VF/NER/TE/NC/C)\n(VF) Varifocal, (NER) Named Entity Recognition, (TE) Triple Extraction, (NC) Naked Claim, (C) Combined\n")
        output_dir = os.path.join(current_dir, '..', 'data', 'analysis')

        skeleton(database_path, method, output_dir)
    elif load_or_analyze == "p":
        data_analysis_path = os.path.join(current_dir, '..', 'data', 'analysis')
        relationship_plotter(data_analysis_path)
    else:
        print("Invalid input. Please enter 'l' or 'a'.")
