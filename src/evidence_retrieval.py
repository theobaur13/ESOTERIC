import json
from models import Evidence

class EvidenceRetriever:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self.load_data(dataset_path)

    def load_data(self, dataset_path):
        data = []
        with open(dataset_path, 'r') as file:
            data = json.load(file)
        return data

    def retrieve_evidence(self, claim):
        evidence = self.data
        score = 1
        return Evidence(claim, evidence, score)