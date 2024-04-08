import os
import json
import logging
import warnings
from transformers.utils import logging as transformers_logging
from analysis.retrieval_analysis import skeleton
from analysis.retrieval_stats import method_performance, plot_performance

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
        analyse_results(output_dir)
    else:
        print("Invalid input. Please enter 'r' or 'a'.")

def run_analysis(current_dir, output_dir):
    database_path = os.path.join(current_dir, '..', 'data')

    preloaded_claim = input("Enter a claim ID to analyze, or type any character to analyze all claims:\n")

    if preloaded_claim.isdigit():
        preloaded_claim = int(preloaded_claim)
        skeleton(database_path, output_dir, preloaded_claim)
    else:
        skeleton(database_path, output_dir)

def analyse_results(output_dir):
    data = load_data(output_dir, "retrieval_results.json")

    # method_performance(data)
    plot_performance(data)

if __name__ == "__main__":
    main()