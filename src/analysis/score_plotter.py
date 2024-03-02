import os
import json
import matplotlib.pyplot as plt

def load_data(output_dir, file_name):
    # Load the data from the json file
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'r') as file:
        data = json.load(file)

    return data

def plot_text_match_scores(data):
    # x-axis: length of evidence set, y-axis: score
    hit_scores = []
    evidence_count = []

    for claim in data:
        for evidence in claim["evidence"]:
            if evidence["hit_status"]:
                hit_scores.append(float(evidence["score"]))
                evidence_count.append(len([e for e in claim["evidence"] if e["entity"] == evidence["entity"]]))

    # Plot the data in a scatter plot
    plt.scatter(evidence_count, hit_scores)
    plt.xlabel("Number of pieces of evidence")
    plt.ylabel("Hit score")
    plt.title("Number of pieces of evidence vs hit score")
    plt.show()

def plot_hit_position(data):
    # x-axis: length of evidence set, y-axis: position of hit
    hit_positions = []
    evidence_count = []

    for claim in data:
        for i, evidence in enumerate(claim["evidence"]):
            if evidence["hit_status"]:
                hit_positions.append(i)
                evidence_count.append(len([e for e in claim["evidence"] if e["entity"] == evidence["entity"]]))

    # Plot the data in a scatter plot
    plt.scatter(evidence_count, hit_positions)
    plt.xlabel("Number of pieces of evidence")
    plt.ylabel("Position of hit")
    plt.title("Number of pieces of evidence vs position of hit")
    plt.show()

def plot_hit_relative_position(data):
    # x-axis: score range, y-axis: hit score / score range
    hit_positions  = []
    score_ranges = []

    for claim in data:
        entities = [e["entity"] for e in claim["evidence"] if e["hit_status"]]
        evidence_count_per_entity = {entity: len([e for e in claim["evidence"] if e["entity"] == entity]) for entity in entities}
        for evidence in claim["evidence"]:
            if evidence["entity"] in entities and evidence["hit_status"] and evidence_count_per_entity[evidence["entity"]] > 1:
                hit_score = float(evidence["score"])
                max_score = max([float(e["score"]) for e in claim["evidence"] if e["entity"] == evidence["entity"]])
                min_score = min([float(e["score"]) for e in claim["evidence"] if e["entity"] == evidence["entity"]])
                score_range = max_score - min_score

                hit_positions.append((hit_score - min_score) / score_range)
                score_ranges.append(score_range)

    # Plot the data in a scatter plot
    plt.scatter(score_ranges, hit_positions)
    plt.xlabel("Score range")
    plt.ylabel("Hit score / score range")
    plt.title("Score range vs hit score / score range")
    plt.show()