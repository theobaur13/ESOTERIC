import sqlite3
import re
import json
import os
import random
from tqdm import tqdm

def create_dataset(database_path, output_dir, limit=10000, x=1, y=1):
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
        cursor.execute("SELECT lines FROM documents WHERE doc_id = ?", (row["doc_id"],))
        row["lines"] = cursor.fetchone()

    # Remove rows where the lines are None
    rows = [row for row in rows if row["lines"] is not None]

    dataset = {"claims": []}
    for row in tqdm(rows):
        claim = row["claim"]
        text = row["lines"][0]
        sent_id = row["sent_id"]
        doc_id = row["doc_id"]

        pattern = re.compile(r'\n\d+\t')
        sentences = pattern.split(text)
        relevant_sentence = sentences[sent_id]

        if claim not in [d['claim'] for d in dataset['claims']]:
            dataset['claims'].append({"claim": claim, "sentences": []})

        for row in dataset['claims']:
            if claim == row['claim']:
                row['sentences'].append({"doc_id": doc_id, "sent_id": sent_id, "sentence": relevant_sentence, "label": 1})

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
                    SELECT doc_id, lines
                    FROM documents
                    WHERE doc_id IN ({})
                    """.format(','.join('?' * len(relevant_sentences_ids))), [d['doc_id'] for d in relevant_sentences_ids])
        for doc in cursor.fetchall():
            documents.append({"doc_id": doc[0], "lines": doc[1]})

        # pre_select_limit = 100
        # placeholders = ','.join('?' * len(relevant_sentences_ids))
        
        # sql_query = f"""
        #     SELECT doc_id, lines
        #     FROM (
        #         SELECT doc_id, lines
        #         FROM documents
        #         WHERE doc_id NOT IN ({placeholders})
        #         LIMIT {pre_select_limit}  -- Pre-select a reasonable number of documents
        #     ) AS pre_selected
        #     ORDER BY RANDOM()
        #     LIMIT ?;
        # """

        # # Select different_doc_irrelevant_sentence_count documents that are not listed in the doc_id column relevant_sentences_ids
        # parameters = [d['doc_id'] for d in relevant_sentences_ids] + [different_doc_irrelevant_sentence_count]
        # cursor.execute(sql_query, parameters)
        # for doc in cursor.fetchall():
        #     documents.append({"doc_id": doc[0], "lines": doc[1]})
            
        # Find max id column in documents
        cursor.execute("SELECT MAX(id) FROM documents")
        max_id = cursor.fetchone()[0]

        # Pick a random id from the documents table, ensuring that it is not in the relevant_sentences_ids doc_id column
        for i in range(different_doc_irrelevant_sentence_count):
            random_id = random.randint(1, max_id)

            found_different = False
            while not found_different:
                cursor.execute("SELECT doc_id, lines FROM documents WHERE id = ?", (random_id,))
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