import sqlite3
import re
import json
import os
import random
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

def create_dataset(database_path, output_dir, limit=10000, x=2, y=2):
    ouput_file = os.path.join(output_dir, 'relevancy_classification.json')

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Select the id of limit number of claims
    cursor.execute("SELECT claim_id FROM claims LIMIT ?", (limit,))
    claim_ids = cursor.fetchall()

    rows = []
    for claim_id in tqdm(claim_ids):
        cursor.execute("""
            SELECT c.claim, cd.sent_id, cd.doc_id
            FROM claims c
            JOIN claim_docs cd ON c.claim_id = cd.claim_id
            WHERE c.claim_id = ?
            """, (claim_id[0],))
        for row in cursor.fetchall():
            rows.append({"claim": row[0], "sent_id": row[1], "doc_id": row[2]})

    # Select the lines at each doc_id and append them to the rows list
    for row in tqdm(rows):
        cursor.execute("SELECT text FROM documents WHERE doc_id = ?", (row["doc_id"],))
        row["lines"] = cursor.fetchone()

    # Remove rows where the lines are None
    rows = [row for row in rows if row["lines"] is not None]

    dataset = {"claims": []}
    claims_lookup= {}
    for row in tqdm(rows):
        claim = row["claim"]
        text = row["lines"][0]
        sent_id = row["sent_id"]
        doc_id = row["doc_id"]

        pattern = re.compile(r'\n\d+\t')
        sentences = pattern.split(text)
        relevant_sentence = sentences[sent_id]
        
        if claim not in claims_lookup:
            new_claim_entry = {"claim": claim, "sentences": [{"doc_id": doc_id, "sent_id": sent_id, "sentence": relevant_sentence, "label": 1}]}
            dataset['claims'].append(new_claim_entry)
            claims_lookup[claim] = new_claim_entry["sentences"]

        else:
            claims_lookup[claim].append({"doc_id": doc_id, "sent_id": sent_id, "sentence": relevant_sentence, "label": 1})

    for row in tqdm(dataset['claims']):
        relevent_sentence_count = len(row['sentences'])
        same_doc_irrelevant_sentence_count = x * relevent_sentence_count
        different_doc_irrelevant_sentence_count = y * relevent_sentence_count

        relevant_sentences_ids = []
        for sentence in row['sentences']:
            relevant_sentences_ids.append({"doc_id": sentence['doc_id'], "sent_id": sentence['sent_id']})

        documents = []
        # Select documents that are listed in the doc_id column relevant_sentences_ids
        cursor.execute("""
                    SELECT doc_id, text
                    FROM documents
                    WHERE doc_id IN ({})
                    """.format(','.join('?' * len(relevant_sentences_ids))), [d['doc_id'] for d in relevant_sentences_ids])
        for doc in cursor.fetchall():
            documents.append({"doc_id": doc[0], "lines": doc[1]})
            
        # Find max id column in documents
        cursor.execute("SELECT MAX(id) FROM documents")
        max_id = cursor.fetchone()[0]

        # Pick a random id from the documents table, ensuring that it is not in the relevant_sentences_ids doc_id column
        for i in range(different_doc_irrelevant_sentence_count):
            random_id = random.randint(1, max_id)

            found_different = False
            while not found_different:
                cursor.execute("SELECT doc_id, text FROM documents WHERE id = ?", (random_id,))
                doc = cursor.fetchone()
                doc_id = doc[0]
                text = doc[1]

                if doc_id not in [d['doc_id'] for d in relevant_sentences_ids]:
                    found_different = True
            documents.append({"doc_id": doc_id, "lines": text})

        # Select same_doc_irrelevant_sentence_count sentences from the same document as the relevant sentences randomly
        same_doc_irrelevant_sentences = []
        different_doc_irrelevant_sentences = []
        for doc in documents:
            pattern = re.compile(r'\n\d+\t')
            sentences = pattern.split(doc['lines'])
            if doc['doc_id'] in [d['doc_id'] for d in relevant_sentences_ids]:
                for sentence in sentences:
                    sent_id = sentences.index(sentence)
                    if sent_id not in [d['sent_id'] for d in relevant_sentences_ids] and sentence != "":
                        same_doc_irrelevant_sentences.append({"doc_id": doc['doc_id'], "sent_id": sent_id, "sentence": sentence, "label": 0})
            else:
                for sentence in sentences:
                    if sentence != "":
                        different_doc_irrelevant_sentences.append({"doc_id": doc['doc_id'], "sent_id": sentences.index(sentence), "sentence": sentence, "label": 0})

        # Shuffle same_doc_irrelevant_sentences and different_doc_irrelevant_sentences
        random.shuffle(same_doc_irrelevant_sentences)
        random.shuffle(different_doc_irrelevant_sentences)

        # Cut off the lists at the same_doc_irrelevant_sentence_count and different_doc_irrelevant_sentence_count
        same_doc_irrelevant_sentences = same_doc_irrelevant_sentences[:same_doc_irrelevant_sentence_count]
        different_doc_irrelevant_sentences = different_doc_irrelevant_sentences[:different_doc_irrelevant_sentence_count]

        # Add the relevant sentences and the same_doc_irrelevant_sentences and different_doc_irrelevant_sentences to the dataset
        row['sentences'] += same_doc_irrelevant_sentences
        row['sentences'] += different_doc_irrelevant_sentences

    with open(ouput_file, 'w') as f:
        json.dump(dataset, f, indent=4)

def train_model(dataset_file, model_name, output_dir):
    with open(dataset_file) as f:
        dataset = json.load(f)
        print("Dataset loaded successfully")

    claims = dataset['claims']
    sentences = []
    labels = []

    for claim in claims:
        for sentence in claim['sentences']:
            concatenated_text = f"{claim['claim']} [SEP] {sentence['sentence']}"
            sentences.append(concatenated_text)
            labels.append(sentence['label'])

    train_texts, test_texts, train_labels, test_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(test_texts, test_labels, test_size=0.5, random_state=42)

    print(f"Training texts: {len(train_texts)}, Training labels: {len(train_labels)}")
    print(f"Validation texts: {len(val_texts)}, Validation labels: {len(val_labels)}")
    print(f"Test texts: {len(test_texts)}, Test labels: {len(test_labels)}")

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
    print("Tokenization complete")

    class RelevancyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
        
    train_dataset = RelevancyDataset(train_encodings, train_labels)
    val_dataset = RelevancyDataset(val_encodings, val_labels)
    test_dataset = RelevancyDataset(test_encodings, test_labels)

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    logging_dir = os.path.join(output_dir, 'logs')

    # Parameters designed for colab machine
    training_args = TrainingArguments(
        output_dir=output_dir,           # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=24,  # batch size per device during training
        per_device_eval_batch_size=30,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=logging_dir,         # directory for storing logs
        evaluation_strategy="epoch",
        save_total_limit=8,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
    )

    print("Starting model training")
    trainer.train(resume_from_checkpoint=True)
    print("Model training complete")
    trainer.evaluate(test_dataset)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully")
    return model, tokenizer

def evaluate_model(dataset_file, model_name):
    with open(dataset_file) as f:
        dataset = json.load(f)
        print("Dataset loaded successfully")

    claims = dataset['claims']
    sentences = []
    labels = []

    for claim in claims:
        for sentence in claim['sentences']:
            concatenated_text = f"{claim['claim']} [SEP] {sentence['sentence']}"
            sentences.append(concatenated_text)
            labels.append(sentence['label'])

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=512)

    class RelevancyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
        
    dataset = RelevancyDataset(encodings, labels)

    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    trainer = Trainer(model=model)
    trainer.evaluate(dataset)