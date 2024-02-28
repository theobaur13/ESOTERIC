import os
import json
import venn
import matplotlib.pyplot as plt

def relationship_plotter(path):
    data = []

    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(os.path.join(path, file), 'r') as f:
                data.append(json.load(f))

    # Sets are NC, NER, TE, VF with data are the claims that are hits
    NC = set()
    NER = set()
    TE = set()
    VF = set()

    for result in data:
        for entry in result:
            if entry["method"] == "NC":
                if entry["hit/miss"] == "hit":
                    NC.add(entry["claim"])
            elif entry["method"] == "NER":
                if entry["hit/miss"] == "hit":
                    NER.add(entry["claim"])
            elif entry["method"] == "TE":
                if entry["hit/miss"] == "hit":
                    TE.add(entry["claim"])
            elif entry["method"] == "VF":
                if entry["hit/miss"] == "hit":
                    VF.add(entry["claim"])

    labels = venn.get_labels([NC, NER, TE, VF], fill=['number'])
    fig, ax = venn.venn4(labels, names=['NC', 'NER', 'TE', 'VF'])
    plt.show()