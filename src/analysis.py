import os
import json
import logging
import warnings
from transformers.utils import logging as transformers_logging
from analysis.retrieval_analysis import skeleton
from analysis.score_analysis import text_match_scoring, disambig_scoring
from analysis.score_plotter import plot_text_match_scores, plot_hit_position, plot_hit_relative_position
from analysis.retrieval_stats import method_performance

# Configure logging and warnings
logging.basicConfig(level=logging.WARNING)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def load_data(output_dir, file_name):
    # Load the data from the json file
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'r') as file:
        data = json.load(file)

    return data

def handle_input():
    operation = input("Would you like to run system or analyse results?\n(r) Run, (a) Analyse\n")
    return operation.lower()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'data', 'analysis')

    operation = handle_input()
    if operation == "r":
        run_analysis(current_dir, output_dir)
    elif operation == "a":
        analyse_results(current_dir)
    else:
        print("Invalid input. Please enter 'r' or 'a'.")

def run_analysis(current_dir, output_dir):
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

def analyse_results(output_dir):
    data = load_data(output_dir, "retrieval_results.json")

    score_or_retrieval = input("Would you like to analyze retrieval performance or scoring?\n(r) Retrieval, (s) Scoring\n")
    if score_or_retrieval == "r":
        method_performance(data)
    elif score_or_retrieval == "s":
        plot_text_match_scores(data)
        plot_hit_position(data)
        plot_hit_relative_position(data)
    else:
        print("Invalid input. Please enter 'r' or 's'.")

if __name__ == "__main__":
    main()