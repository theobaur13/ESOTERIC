import os
import json

def relationship_plotter(path):
    for file in os.listdir(path):
        if file.endswith(".json"):
            with open(os.path.join(path, file), 'r') as f:
                data = json.load(f)
                