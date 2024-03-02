import logging
logging.basicConfig(level=logging.WARNING)

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

import os
from analysis.retrieval_loader import retrieval_loader
from analysis.retrieval_analysis import skeleton
from analysis.score_analysis import text_match_scoring, disambig_scoring
from analysis.score_plotter import load_data, plot_text_match_scores, plot_hit_position, plot_hit_relative_position

if __name__ == "__main__":
    load_or_analyze = input("Would you like to load the data, analyze it, or plot it?\n(l) Load, (a) Analyze, (p) Plot\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'data', 'analysis')

    if load_or_analyze == "l":
        retrieval_loader()

    elif load_or_analyze == "a":
        database_path = os.path.join(current_dir, '..', 'data')

        score_or_retrieval = input("Would you like to analyze retrieval performance or scoring?\n(r) Retrieval, (s) Scoring\n")
        preloaded_claim = input("Enter a claim ID to analyze, or type any character to analyze all claims:\n")

        if score_or_retrieval == "r":
            if preloaded_claim.isdigit():
                preloaded_claim = int(preloaded_claim)

                skeleton(database_path, output_dir, preloaded_claim)

            else:
                skeleton(database_path, output_dir)

        elif score_or_retrieval == "s":
            disambig_or_text = input("Would you like to analyze disambiguation or text matching?\n(d) Disambiguation, (t) Text matching\n")
            if disambig_or_text == "d":
                if preloaded_claim.isdigit():
                    preloaded_claim = int(preloaded_claim)

                    disambig_scoring(database_path, output_dir, preloaded_claim)

                else:
                    disambig_scoring(database_path, output_dir)

            elif disambig_or_text == "t":
                if preloaded_claim.isdigit():
                    preloaded_claim = int(preloaded_claim)

                    text_match_scoring(database_path, output_dir, preloaded_claim)

                else:
                    text_match_scoring(database_path, output_dir)

    elif load_or_analyze == "p":
        data = load_data(output_dir, "disambiguation_score_results.json")
        plot_text_match_scores(data)
        plot_hit_position(data)
        plot_hit_relative_position(data)

    else:
        print("Invalid input. Please enter 'l' or 'a'.")