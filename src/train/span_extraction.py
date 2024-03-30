import os
import json
import sqlite3
from tqdm import tqdm
from evidence_retrieval.tools.NER import extract_entities
from transformers import pipeline

def create_span_dataset(database_path, output_dir, limit=10000):
    answer_extraction_pipe = pipeline("text2text-generation", model="vabatista/t5-small-answer-extraction-en")
    NER_model = pipeline("token-classification", model="Babelscape/wikineural-multilingual-ner", grouped_entities=True)

    ouput_file = os.path.join(output_dir, 'span_extraction_' + str(limit) + '.json')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT claim FROM claims LIMIT ?", (limit,))
    rows = cursor.fetchall()

    dataset = {"data": []}
    for row in tqdm(rows):
        claim = row[0]
        entities = extract_entities(answer_extraction_pipe, NER_model, claim)
        dataset["data"].append({"text": claim, "entities": entities})

    with open(ouput_file, 'w') as f:
        json.dump(dataset, f)

def train_span_model(dataset_file, model_name, output_dir):
    pass