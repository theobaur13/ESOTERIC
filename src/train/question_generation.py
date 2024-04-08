import os
import sqlite3
import json
from transformers import pipeline
from tqdm import tqdm
from evidence_retrieval.tools.document_retrieval import extract_answers, extract_questions

def create_question_dataset(database_path, output_dir, limit=2000):
    ouput_file = os.path.join(output_dir, 'question_generation_' + str(limit) + '.json')
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    answer_extraction_pipe = pipeline("text2text-generation", model="vabatista/t5-small-answer-extraction-en")
    question_generation_pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-question-generation-ap", max_length=256)

    cursor.execute("SELECT claim FROM claims LIMIT ?", (limit,))

    dataset = {"data": []}
    for row in tqdm(cursor.fetchall()):
        claim = row[0]
        return_dict = {"context": claim, "questions": []}

        answers = extract_answers(answer_extraction_pipe, claim)
        for answer in answers:
            question = extract_questions(question_generation_pipe, answer["focal"], claim)
            return_dict["questions"].append({"answer": answer["focal"], "question": question})
        dataset["data"].append(return_dict)
            
    with open(ouput_file, 'w') as f:
        json.dump(dataset, f)

def train_question_model(dataset_file, model_name, output_dir):
    pass