import os
import sqlite3
import json
import re
from tqdm import tqdm
import time
import unicodedata
import random
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from legacy.FEVERISH_1.FEVERISH_1 import EvidenceRetriever as EvidenceRetriever_1
from legacy.FEVERISH_3.FEVERISH_3 import EvidenceRetriever as EvidenceRetriever_3
from legacy.FEVERISH_3_1.FEVERISH_3_1 import EvidenceRetriever as EvidenceRetriever_3_1
from legacy.FEVERISH_3_2.FEVERISH_3_2 import EvidenceRetriever as EvidenceRetriever_3_2
from legacy.FEVERISH_3_3.FEVERISH_3_3 import EvidenceRetriever as EvidenceRetriever_3_3
from legacy.FEVERISH_3_4.FEVERISH_3_4 import EvidenceRetriever as EvidenceRetriever_3_4
from legacy.FEVERISH_3_5.FEVERISH_3_5 import EvidenceRetriever as EvidenceRetriever_3_5
from legacy.FEVERISH_3_6.FEVERISH_3_6 import EvidenceRetriever as EvidenceRetriever_3_6
from legacy.FEVERISH_3_7.FEVERISH_3_7 import EvidenceRetriever as EvidenceRetriever_3_7
from legacy.final_version.final_version import EvidenceRetriever as EvidenceRetriever_Final

def print_scores(doc_hits=0, doc_misses=0, doc_targets=0, passage_hits=0, passage_misses=0, passage_targets=0, combined_hits=0, combined_misses=0, combined_targets=0, FEVER_doc_hits=0, FEVER_passage_hits=0, FEVER_combined_hits=0, execution_avg=0, record_count=0):
    def safe_division(numerator, denominator):
        if denominator == 0:
            return 0
        return numerator / denominator

    print("Precision (doc): " + str(safe_division(doc_hits, (doc_hits + doc_misses)) * 100) + "%")
    print("Recall (doc): " + str(safe_division(doc_hits, doc_targets) * 100) + "%")
    print("Precision (passage): " + str(safe_division(passage_hits, (passage_hits + passage_misses)) * 100) + "%")
    print("Recall (passage): " + str(safe_division(passage_hits, passage_targets) * 100) + "%")
    print("Precision (combined): " + str(safe_division(combined_hits, (combined_hits + combined_misses)) * 100) + "%")
    print("Recall (combined): " + str(safe_division(combined_hits, combined_targets) * 100) + "%")
    print("FEVER doc score: " + str(safe_division(FEVER_doc_hits, record_count) * 100) + "%")
    print("FEVER passage score: " + str(safe_division(FEVER_passage_hits, record_count) * 100) + "%")
    print("FEVER combined score: " + str(safe_division(FEVER_combined_hits, record_count) * 100) + "%")
    print("Execution average: " + str(execution_avg) + "s")

"""
{
    "claim": "claim_text",
    "target": [
            {
                "doc_id": "document_id",
                "sent_id": "0",
                "span_start": 0,
                "span_end": 0
            }
        ],
        "evidence": [
            {
                "doc_id": "document_id",
                "sentences": [
                    {
                        "sentence": "0",
                        "score": 0,
                        "start": 0,
                        "end": 0,
                        "method": "method"
                    }
                ],
                "score": 0,
                "method": "method",
                "doc_hit": False,
                "sentence_hit": False
            }
        ]
}
"""

def get_claims(seed, conn, batch_limit=200):
    cursor = conn.cursor()
    random.seed(seed)
    file_ids = random.sample(range(1, 110), 5)
    print(file_ids)

    # get all claims with doc_ids in file_ids
    cursor.execute('''
        SELECT claim_id
        FROM claim_docs
        WHERE doc_no IN ({})
    '''.format(','.join(['?']*len(file_ids))), tuple(file_ids))
    claim_ids = [row[0] for row in cursor.fetchall()]

    # fetch a random sample of claim ids from the claim_ids
    random_claim_ids = random.sample(claim_ids, batch_limit)

    # get the claims and doc_sent_pairs for the random sample of claim ids
    cursor.execute('''
        SELECT c.claim, GROUP_CONCAT(cd.doc_id || ':' || cd.sent_id, ';') AS doc_sent_pairs
        FROM claims c
        JOIN claim_docs cd ON c.claim_id = cd.claim_id
        WHERE c.claim_id IN ({})
        GROUP BY c.claim_id
    '''.format(','.join('?' * len(random_claim_ids))), random_claim_ids)
    rows = cursor.fetchall()
    return rows

def write_to_file(output_file, data):
    with open(output_file, 'w') as file:
        json.dump(data, file)

def run_without_passage(output_file, retriever_model, claim_db_path, wiki_db_path, batch_limit, seed):
    retriever_models = {
        "FEVERISH_1": EvidenceRetriever_1,
        "FEVERISH_3": EvidenceRetriever_3,
        "FEVERISH_3_1": EvidenceRetriever_3_1,
        "FEVERISH_3_2": EvidenceRetriever_3_2,
        "FEVERISH_3_3": EvidenceRetriever_3_3,
    }

    conn = sqlite3.connect(os.path.join(claim_db_path, 'data.db'))
    rows = get_claims(seed, conn, batch_limit)

    evidence_retriever = retriever_models[retriever_model](wiki_db_path)

    if not os.path.exists(output_file):
        with open(output_file, 'w') as file:
            json.dump([], file)

    with open(output_file, 'r') as file:
        data = json.load(file)

    doc_hits = 0
    doc_misses = 0
    doc_targets = 0

    FEVER_doc_hits = 0

    execution_avg = 0
    record_count = 0

    for record in data:
        record_count += 1
        execution_time = record["execution_time"]
        execution_avg = (execution_avg * record_count + execution_time) / (record_count + 1)

        target_docs = []
        evidence_docs = [evidence["doc_id"] for evidence in record["evidence"]]
        
        # Set target documents
        for target in record["target"]:
            if target["doc_id"] not in target_docs:
                target_docs.append(target["doc_id"])
        print(target_docs)
        # Set counters for targets
        doc_targets += len(target_docs)

        hitted_docs = []
        for evidence in record["evidence"]:
            if evidence["doc_id"] in target_docs and evidence["doc_id"] not in hitted_docs:
                doc_hits += 1
                hitted_docs.append(evidence["doc_id"])
            else:
                doc_misses += 1

        # if target_docs are all in evidence_docs, then it's a FEVER doc hit
        if set(target_docs).issubset(evidence_docs):
            FEVER_doc_hits += 1

        print_scores(doc_hits=doc_hits, doc_misses=doc_misses, doc_targets=doc_targets, FEVER_doc_hits=FEVER_doc_hits, execution_avg=execution_avg, record_count=record_count)

    for row in tqdm(rows):
        # Check if the claim is already in the output file, if so, skip
        claim = row[0]
        if any(record["claim"] == claim for record in data):
            print("Claim already in output file, skipping...")
            continue

        doc_sent_pairs = row[1].split(';')
        # target_docs = list(set([doc_sent_pair.split(':')[0] for doc_sent_pair in doc_sent_pairs]))
        target_docs = []
        for pair in doc_sent_pairs:
            doc_id = pair.split(':')[0]
            if doc_id not in target_docs:
                target_docs.append(doc_id)

        start_time = time.time()
        evidence_wrapper = evidence_retriever.retrieve_evidence(claim)
        execution_time = time.time() - start_time

        # Update counters for targets
        doc_targets += len(target_docs)

        # Create the record
        record = {
            "claim": claim,
            "target": [],
            "evidence": [],
            "execution_time": execution_time
        }

        # Add target documents to the result dictionary
        for pair in doc_sent_pairs:
            doc_id, sent_id = pair.split(':')
            record["target"].append({
                "doc_id": doc_id,
                "sent_id": sent_id,
                "span_start": 0,
                "span_end": 0
            })
        hitted_docs = []
        for evidence in evidence_wrapper.get_evidences():
            doc_hit = False

            if evidence.doc_id == doc_id:
                doc_hit = True
                if evidence.doc_id not in hitted_docs:
                    doc_hits += 1
                    hitted_docs.append(doc_id)
            else:
                doc_misses += 1

            to_append = {
                "doc_id": evidence.doc_id,
                "sentences": [],
                "score": float(evidence.doc_score),
                "method": evidence.doc_retrieval_method,
                "doc_hit": doc_hit,
                "sentence_hit": False
            }
            record["evidence"].append(to_append)

        evidence_docs = [evidence["doc_id"] for evidence in record["evidence"]]

        # if target_docs are all in evidence_docs, then it's a FEVER doc hit
        if set(target_docs).issubset(evidence_docs):
            FEVER_doc_hits += 1

        with open(output_file, 'r') as file:
            data = json.load(file)
        
        data.append(record)

        with open(output_file, 'w') as file:
            json.dump(data, file)

        print_scores(doc_hits=doc_hits, doc_misses=doc_misses, doc_targets=doc_targets, FEVER_doc_hits=FEVER_doc_hits, execution_avg=execution_avg, record_count=record_count)

def run_with_passage(output_file, retriever_model, claim_db_path, wiki_db_path, batch_limit, seed, flush_questions=False, normalize=False):
    retriever_models = {
        "FEVERISH_1": EvidenceRetriever_1,
        "FEVERISH_3": EvidenceRetriever_3,
        "FEVERISH_3_1": EvidenceRetriever_3_1,
        "FEVERISH_3_2": EvidenceRetriever_3_2,
        "FEVERISH_3_3": EvidenceRetriever_3_3,
        "FEVERISH_3_4": EvidenceRetriever_3_4,
        "FEVERISH_3_5": EvidenceRetriever_3_5,
        "FEVERISH_3_6": EvidenceRetriever_3_6,
        "FEVERISH_3_7": EvidenceRetriever_3_7,
        "final_version": EvidenceRetriever_Final
    }

    conn = sqlite3.connect(os.path.join(claim_db_path, 'data.db'))
    rows = get_claims(seed, conn, batch_limit)

    if retriever_model in ["FEVERISH_1", "FEVERISH_3", "FEVERISH_3_1", "FEVERISH_3_2", "FEVERISH_3_3", "FEVERISH_3_4", "FEVERISH_3_5"]:
        evidence_retriever = retriever_models[retriever_model](wiki_db_path)
    if retriever_model in ["FEVERISH_3_6", "FEVERISH_3_7", "final_version"]:
        evidence_retriever = retriever_models[retriever_model](title_match_docs_limit=1000)

    if not os.path.exists(output_file):
        with open(output_file, 'w') as file:
            json.dump([], file)

    with open(output_file, 'r') as file:
        data = json.load(file)

    doc_hits = 0
    doc_misses = 0
    doc_targets = 0

    passage_hits = 0
    passage_misses = 0
    passage_targets = 0

    combined_hits = 0
    combined_misses = 0
    combined_targets = 0

    FEVER_doc_hits = 0
    FEVER_passage_hits = 0
    FEVER_combined_hits = 0

    execution_avg = 0
    record_count = 0

    for record in data:
        record_count += 1
        execution_time = record["execution_time"]
        execution_avg = (execution_avg * record_count + execution_time) / (record_count + 1)

        target_docs = []
        target_passages = []
        target_combined = []
        evidence_docs = [evidence["doc_id"] for evidence in record["evidence"]]

        # Set target documents
        for target in record["target"]:
            if target["doc_id"] not in target_docs:
                target_docs.append(target["doc_id"])

        # Set target passages
        for target in record["target"]:
            target_combined.append((target["doc_id"], (target["span_start"], target["span_end"])))
            if target["doc_id"] in evidence_docs:
                target_passages.append((target["doc_id"], (target["span_start"], target["span_end"])))

        # Set counters for targets
        doc_targets += len(target_docs)
        passage_targets += len(target_passages)
        combined_targets += len(target_combined)

        evidence_passages = []
        for evidence in record["evidence"]:
            for sentence in evidence["sentences"]:
                evidence_passages.append((evidence["doc_id"], (sentence["start"], sentence["end"])))

            doc_hit = False
            passage_hit = False
            combined_hit = False

            if evidence["doc_id"] in target_docs:
                doc_hit = True

            for sentence in evidence["sentences"]:
                for target in target_passages:
                    if type(target[1][0]) == int and type(target[1][1]) == int and (sentence["start"] < target[1][1]) and (sentence["end"] > target[1][0]) and (evidence["doc_id"] == target[0]):
                        if (sentence["start"] < target[1][1]) and (sentence["end"] > target[1][0]) and (evidence["doc_id"] == target[0]):
                            passage_hit = True

                if doc_hit and passage_hit:
                    combined_hit = True

            if doc_hit:
                doc_hits += 1
            else:
                doc_misses += 1
            if passage_hit:
                passage_hits += 1
            else:
                if doc_hit:
                    passage_misses += 1
            if combined_hit:
                combined_hits += 1
            else:
                combined_misses += 14

        # if target_docs are all in evidence_docs, then it's a FEVER doc hit
        if set(target_docs).issubset(evidence_docs):
            FEVER_doc_hits += 1

        # if target_passages are all in evidence_passages, then it's a FEVER passage hit
        if subset_checker(target_passages, evidence_passages):
            FEVER_passage_hits += 1

        if subset_checker(target_combined, evidence_passages):
            FEVER_combined_hits += 1

        print_scores(doc_hits=doc_hits, doc_misses=doc_misses, doc_targets=doc_targets, passage_hits=passage_hits, passage_misses=passage_misses, passage_targets=passage_targets, combined_hits=combined_hits, combined_misses=combined_misses, combined_targets=combined_targets, FEVER_doc_hits=FEVER_doc_hits, FEVER_passage_hits=FEVER_passage_hits, FEVER_combined_hits=FEVER_combined_hits, execution_avg=execution_avg, record_count=record_count)

    cursor = conn.cursor()

    for row in tqdm(rows):
        # Check if the claim is already in the output file, if so, skip
        claim = row[0]
        if any(record["claim"] == claim for record in data):
            print("Claim already in output file, skipping...")
            continue

        record_count += 1
        evidence_pairs = row[1].split(';')

        target_docs = list(set([pair.split(':')[0] for pair in evidence_pairs]))
        print("\nTarget documents:", target_docs)

        # Run the evidence retrieval
        start = time.time()
        if flush_questions:
            evidence_retriever.flush_questions()
        evidence_wrapper = evidence_retriever.retrieve_evidence(claim)
        end = time.time()
        execution_time = end - start
        print("Execution time:", execution_time)

        # Update counters for execution time
        execution_avg = (execution_avg * record_count + execution_time) / (record_count + 1)

        target_passages = []
        target_combined = []
        evidence_docs = list(set([evidence.doc_id for evidence in evidence_wrapper.get_evidences()]))

        # Set target passages
        for pair in evidence_pairs:
            doc_id = pair.split(':')[0]
            sent_id = pair.split(':')[1]
            target_combined.append((doc_id, sent_id))
            if doc_id in evidence_docs:
                target_passages.append((doc_id, sent_id))

        # Find the span positions of the target sentences in the documents
        target_span_positions_passage = []
        target_span_positions_combined = []
        target_span_positions_combined_map = {}
        for doc_id in target_docs:
            # get text from documents db for doc_id
            cursor.execute('''
                SELECT text
                FROM documents
                WHERE doc_id = ?
            ''', (doc_id,))

            output = cursor.fetchone()
            if output is None:
                lines = ""
            else:
                lines = output[0]
            # lines = cursor.fetchone()[0]
            if normalize:
                text = " ".join(normalize_lines(lines))
                split_lines = normalize_lines(lines)
            else:
                text = lines
                pattern = r'\n\d+\t'
                split_lines = re.split(pattern, text)

            combined_relevant_lines = []
            passage_relevant_lines = []
            combined_relevant_lines_map = {}
            for index, line in enumerate(split_lines):
                if (doc_id, str(index)) in target_combined:
                    combined_relevant_lines.append({"line": line, "index": index})
                    if (doc_id, str(index)) in target_passages:
                        passage_relevant_lines.append({"line": line, "index": index})

            for row in combined_relevant_lines:
                line = row["line"]
                id = row["index"]
                indices = find_substring_indices(text, line)
                if indices:
                    target_span_positions_combined.append({"doc_id": doc_id, "start": indices[0], "end": indices[1]})
                    combined_relevant_lines_map[(doc_id, id)] = (indices[0], indices[1]) 

            for row in passage_relevant_lines:
                line = row["line"]
                id = row["index"]
                indices = find_substring_indices(text, line)
                if indices:
                    target_span_positions_passage.append({"doc_id": doc_id, "start": indices[0], "end": indices[1]})

        # Update counters for targets
        doc_targets += len(target_docs)
        passage_targets += len(target_passages)
        combined_targets += len(target_combined)
        print("Target sentences:", target_passages)
        print("Target combined:", target_combined)

        # Store the results in a dictionary
        result = {
            "claim": claim,
            "target": [],
            "evidence": [],
            "execution_time": end - start
            }
        
        # Add target documents to the result dictionary
        for pair in target_combined:
            doc_id = pair[0]
            sent_id = pair[1]
            span_start = None
            span_end = None

            print(combined_relevant_lines_map)

            if (doc_id, int(sent_id)) in combined_relevant_lines_map:
                span_start = combined_relevant_lines_map[(doc_id, int(sent_id))][0]
                span_end = combined_relevant_lines_map[(doc_id, int(sent_id))][1]

            result["target"].append({
                "doc_id": doc_id,
                "sent_id": sent_id,
                "span_start": span_start,
                "span_end": span_end
            })

        # Check if any documents in the evidence wrapper match the target document
        for evidence in evidence_wrapper.get_evidences():
            doc_hit = False
            sentence_hit = False
            combined_hit = False

            if evidence.doc_id in target_docs:
                doc_hit = True

            for sentence in evidence.sentences:
                # Check if the sentence overlaps with the passages target span
                for target_span in target_span_positions_passage:
                    if (sentence.start < target_span["end"]) and (sentence.end > target_span["start"]) and (evidence.doc_id == target_span["doc_id"]):
                        sentence_hit = True

                # Check if the sentence overlaps with the combined target span
                for target_span in target_span_positions_combined:
                    if (sentence.start < target_span["end"]) and (sentence.end > target_span["start"]) and (evidence.doc_id == target_span["doc_id"]):
                        combined_hit = True

            # Add evidence to the result dictionary
            to_append = {
                "doc_id": evidence.doc_id,
                "sentences": [],
                "score": str(evidence.doc_score),
                "method" : evidence.doc_retrieval_method,
                "doc_hit" : doc_hit,
                "sentence_hit" : sentence_hit
            }

            # Add sentences to the result dictionary
            for sentence in evidence.sentences:
                to_append["sentences"].append({
                    "sentence": sentence.sentence,
                    "score": str(sentence.score),
                    "start": sentence.start,
                    "end": sentence.end,
                    "question": sentence.question,
                    "method": sentence.method
                })
            result["evidence"].append(to_append)

            # Update counters for hits and misses
            if doc_hit:
                doc_hits += 1
            else:
                doc_misses += 1
            if sentence_hit:
                passage_hits += 1
            else:
                if doc_hit:
                    passage_misses += 1
            if combined_hit:
                combined_hits += 1
            else:
                combined_misses += 1

        evidence_docs = [evidence.doc_id for evidence in evidence_wrapper.get_evidences()]
        evidence_passages = []
        for evidence in evidence_wrapper.get_evidences():
            for sentence in evidence.sentences:
                evidence_passages.append((evidence.doc_id, (sentence.start, sentence.end)))

        target_passages_set = [(item["doc_id"], (item["start"], item["end"])) for item in target_span_positions_passage]
        target_combined_set = [(item["doc_id"], (item["start"], item["end"])) for item in target_span_positions_combined]

        # if target_docs are all in evidence_docs, then it's a FEVER doc hit
        if set(target_docs).issubset(evidence_docs):
            FEVER_doc_hits += 1

        # if target_passages are all in evidence_passages, then it's a FEVER passage hit
        if subset_checker(target_passages_set, evidence_passages):
            FEVER_passage_hits += 1

        # if target_combined are all in evidence_passages, then it's a FEVER combined hit
        if subset_checker(target_combined_set, evidence_passages):
            FEVER_combined_hits += 1

        with open(output_file, 'r') as file:
            data = json.load(file)

        data.append(result)

        with open(output_file, 'w') as file:
            json.dump(data, file)

        print_scores(doc_hits=doc_hits, doc_misses=doc_misses, doc_targets=doc_targets, passage_hits=passage_hits, passage_misses=passage_misses, passage_targets=passage_targets, combined_hits=combined_hits, combined_misses=combined_misses, combined_targets=combined_targets, FEVER_doc_hits=FEVER_doc_hits, FEVER_passage_hits=FEVER_passage_hits, FEVER_combined_hits=FEVER_combined_hits, execution_avg=execution_avg, record_count=record_count)

def doc_recall(path):
    with open(path, 'r') as file:
        data = json.load(file)

        doc_hits = 0
        doc_targets = 0

        for record in data:
            target_docs = []
            for target in record["target"]:
                if target["doc_id"] not in target_docs:
                    target_docs.append(target["doc_id"])
            doc_targets += len(target_docs)

            evidence_docs = []
            for evidence in record["evidence"]:
                # if evidence["doc_hit"] and evidence["doc_id"] not in evidence_docs:
                if evidence["doc_id"] in target_docs and evidence["doc_id"] not in evidence_docs:
                    doc_hits += 1
                    evidence_docs.append(evidence["doc_id"])

        return doc_hits/doc_targets

def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))

def doc_precision(path):
    with open(path, 'r') as file:
        data = json.load(file)

        doc_hits = 0
        doc_misses = 0

        for record in data:
            target_docs = []
            for target in record["target"]:
                if target["doc_id"] not in target_docs:
                    target_docs.append(target["doc_id"])

            evidence_docs = []
            for evidence in record["evidence"]:
                if evidence["doc_id"] in target_docs and evidence["doc_id"] not in evidence_docs:
                    doc_hits += 1
                    evidence_docs.append(evidence["doc_id"])
                else:
                    doc_misses += 1

        return doc_hits/(doc_hits + doc_misses)

def passage_recall(path):
    with open(path, 'r') as file:
        data = json.load(file)

        passage_hits = 0
        passage_targets = 0

        for record in tqdm(data):
            evidence_docs = [evidence["doc_id"] for evidence in record["evidence"]]

            target_passages = []
            for target in record["target"]:
                if target["doc_id"] in [id for id in evidence_docs]:
                    target_passages.append((target["doc_id"], (target["span_start"], target["span_end"])))
            passage_targets += len(target_passages)

            target_passages = list(set(target_passages))
            target_passages = [target for target in target_passages if type(target[1][0]) == int and type(target[1][1]) == int]
            target_passages = [target for target in target_passages if target[1][0] >= 0 and target[1][1] >= 0]

            for evidence in record["evidence"]:
                if evidence["doc_id"] in [id for id in evidence_docs]:
                    for sentence in evidence["sentences"]:
                        for target in target_passages:
                            if type(sentence["start"]) == int and type(sentence["end"]) == int and type(target[1][0]) == int and type(target[1][1]) == int:
                                if (sentence["start"] < target[1][1]) and (sentence["end"] > target[1][0]) and (evidence["doc_id"] == target[0]):
                                    passage_hits += 1

        return passage_hits/passage_targets

def passage_precision(path):
    with open(path, 'r') as file:
        data = json.load(file)

        passage_hits = 0
        passage_misses = 0

        for record in tqdm(data):
            evidence_docs = [evidence["doc_id"] for evidence in record["evidence"]]

            target_passages = []
            for target in record["target"]:
                if target["doc_id"] in [id for id in evidence_docs]:
                    target_passages.append((target["doc_id"], (target["span_start"], target["span_end"])))

            target_passages = list(set(target_passages))
            target_passages = [target for target in target_passages if type(target[1][0]) == int and type(target[1][1]) == int]
            target_passages = [target for target in target_passages if target[1][0] >= 0 and target[1][1] >= 0]

            for evidence in record["evidence"]:
                if evidence["doc_id"] in [id for id in evidence_docs]:
                    for sentence in evidence["sentences"]:
                        hit = False
                        miss = False
                        for target in target_passages:
                            if evidence["doc_id"] == target[0]:
                                if type(sentence["start"]) == int and type(sentence["end"]) == int and type(target[1][0]) == int and type(target[1][1]) == int:
                                    if (sentence["start"] < target[1][1]) and (sentence["end"] > target[1][0]):
                                        hit = True
                                    else:
                                        miss = True

                        if hit:
                            passage_hits += 1
                        elif miss:
                            passage_misses += 1

        return passage_hits/(passage_hits + passage_misses)

def combined_recall(path):
    with open(path, 'r') as file:
        data = json.load(file)

        combined_hits = 0
        combined_targets = 0

        for record in tqdm(data):
            target_combined = []
            for target in record["target"]:
                if type(target["span_start"]) == int and type(target["span_end"]) == int:
                    if target["span_start"] >= 0 and target["span_end"] >= 0:
                        target_combined.append({"doc_id": target["doc_id"], "span_start": target["span_start"], "span_end": target["span_end"], "hit": False})

            combined_targets += len(target_combined)

            for evidence in record["evidence"]:
                for sentence in evidence["sentences"]:
                    hit = False
                    for target in target_combined:
                        if type(sentence["start"]) == int and type(sentence["end"]) == int and type(target["span_start"]) == int and type(target["span_end"]) == int:
                            if (sentence["start"] < target["span_end"]) and (sentence["end"] > target["span_start"]) and (evidence["doc_id"] == target["doc_id"]):
                                hit = True
                                
                    if hit:
                        combined_hits += 1
        return combined_hits/combined_targets

def combined_precision(path):
    with open(path, 'r') as file:
        data = json.load(file)

        combined_hits = 0
        combined_misses = 0

        for record in tqdm(data):
            target_combined = []
            for target in record["target"]:
                if type(target["span_start"]) == int and type(target["span_end"]) == int:
                    if target["span_start"] >= 0 and target["span_end"] >= 0:
                        target_combined.append((target["doc_id"], (target["span_start"], target["span_end"])))

            target_combined = list(set(target_combined))

            for evidence in record["evidence"]:
                for sentence in evidence["sentences"]:
                    hit = False
                    miss = False
                    for target in target_combined:
                        if type(sentence["start"]) == int and type(sentence["end"]) == int and type(target[1][0]) == int and type(target[1][1]) == int:
                            if (sentence["start"] < target[1][1]) and (sentence["end"] > target[1][0]) and (evidence["doc_id"] == target[0]):
                                hit = True
                            else:
                                miss = True

                    if hit:
                        combined_hits += 1
                    elif miss:
                        combined_misses += 1

        return combined_hits/(combined_hits + combined_misses)

def FEVER_doc_score(path):
    with open(path, 'r') as file:
        data = json.load(file)

        FEVER_doc_hits = 0
        record_count = 0

        for record in data:
            record_count += 1
            target_docs = []
            for target in record["target"]:
                if target["doc_id"] not in target_docs:
                    target_docs.append(target["doc_id"])

            evidence_docs = [evidence["doc_id"] for evidence in record["evidence"]]

            if set(target_docs).issubset(evidence_docs):
                FEVER_doc_hits += 1

        return FEVER_doc_hits/record_count

def FEVER_passage_score(path):
    with open(path, "r") as file:
        data = json.load(file)

        FEVER_passage_hits = 0
        record_count = 0

        for record in data:
            evidence_docs = [evidence["doc_id"] for evidence in record["evidence"]]
            target_passages = []
            for target in record["target"]:
                if target["doc_id"] in evidence_docs:
                    target_passages.append((target["doc_id"], (target["span_start"], target["span_end"])))

            target_passages = list(set(target_passages))
            target_passages = [target for target in target_passages if type(target[1][0]) == int and type(target[1][1]) == int]
            target_passages = [target for target in target_passages if target[1][0] >= 0 and target[1][1] >= 0]

            if target_passages != []:
                record_count += 1

                evidence_passages = []
                for evidence in record["evidence"]:
                    for sentence in evidence["sentences"]:
                        evidence_passages.append((evidence["doc_id"], (sentence["start"], sentence["end"])))

                if subset_checker(target_passages, evidence_passages):
                    FEVER_passage_hits += 1

        return FEVER_passage_hits/record_count

def FEVER_combined_score(path):
    with open(path, 'r') as file:
        data = json.load(file)

        FEVER_combined_hits = 0
        record_count = 0

        for record in data:
            target_combined = []
            for target in record["target"]:
                target_combined.append((target["doc_id"], (target["span_start"], target["span_end"])))

            target_combined = list(set(target_combined))

            if target_combined != []:
                record_count += 1

                evidence_passages = []
                for evidence in record["evidence"]:
                    for sentence in evidence["sentences"]:
                        evidence_passages.append((evidence["doc_id"], (sentence["start"], sentence["end"])))

                if subset_checker(target_combined, evidence_passages):
                    FEVER_combined_hits += 1

        return FEVER_combined_hits/record_count

def execution_avg(path):
    with open(path, 'r') as file:
        data = json.load(file)

        execution_avg = 0
        record_count = 0

        for record in data:
            record_count += 1
            execution_time = record["execution_time"]
            execution_avg = (execution_avg * record_count + execution_time) / (record_count + 1)

        return execution_avg

# How many hits are there for each method?
def method_analysis(path):
    with open(path, 'r') as file:
        data = json.load(file)

        method_hits = {}

        for record in data:
            target_docs = []
            for target in record["target"]:
                if target["doc_id"] not in target_docs:
                    target_docs.append(target["doc_id"])

            for evidence in record["evidence"]:
                method = evidence["method"]
                # if evidence["doc_hit"]:
                if evidence["doc_id"] in target_docs:
                    if method in method_hits:
                        method_hits[method] += 1
                    else:
                        method_hits[method] = 1

        total = sum(method_hits.values())

        for key, value in method_hits.items():
            print(key, ":", (value/total*100), "%")

def method_analysis_passages(path):
    with open(path, 'r') as file:
        data = json.load(file)

        method_hits = {}

        for record in data:
            target_passages = []
            for target in record["target"]:
                target_passages.append((target["doc_id"], (target["span_start"], target["span_end"])))

            for evidence in record["evidence"]:
                for sentence in evidence["sentences"]:
                    method = sentence["method"]
                    for target in target_passages:
                        if type(sentence["start"]) == int and type(sentence["end"]) == int and type(target[1][0]) == int and type(target[1][1]) == int:
                            if (sentence["start"] < target[1][1]) and (sentence["end"] > target[1][0]) and (evidence["doc_id"] == target[0]):
                                if method in method_hits:
                                    method_hits[method] += 1
                                else:
                                    method_hits[method] = 1

        total = sum(method_hits.values())

        print("Passage hits:")
        for key, value in method_hits.items():
            print(key, ":", (value/total*100), "%")

def method_analysis_passages_unique(path):
    with open(path, 'r') as file:
        data = json.load(file)

        method_hits = {}

        for record in data:
            target_passages = []
            for target in record["target"]:
                target_passages.append((target["doc_id"], (target["span_start"], target["span_end"])))

            for evidence in record["evidence"]:
                BM25_hit = False
                FARM_hit = False
                for sentence in evidence["sentences"]:
                    method = sentence["method"]
                    for target in target_passages:
                        if type(sentence["start"]) == int and type(sentence["end"]) == int and type(target[1][0]) == int and type(target[1][1]) == int:
                            if (sentence["start"] < target[1][1]) and (sentence["end"] > target[1][0]) and (evidence["doc_id"] == target[0]):
                                if method == "BM25":
                                    BM25_hit = True
                                if method == "FARM":
                                    FARM_hit = True

                if BM25_hit and FARM_hit:
                    method = "BM25+FARM"
                elif BM25_hit and not FARM_hit:
                    method = "BM25"
                elif FARM_hit and not BM25_hit:
                    method = "FARM"
                else:
                    method = None

                if method:
                    if method in method_hits:
                        method_hits[method] += 1
                    else:
                        method_hits[method] = 1                      
        
        # Create venn diagram
        venn2(subsets = (method_hits["BM25"], method_hits["FARM"], method_hits["BM25+FARM"]), set_labels = ('Lexical', 'Semantic'))
        plt.show()

def average_doc_length(path):
    with open(path, 'r') as file:
        data = json.load(file)

        doc_lengths = []

        for record in data:
            for target in record["target"]:
                doc_id = target["doc_id"]
                doc_lengths.append(len(doc_id))

        return sum(doc_lengths)/len(doc_lengths)

# Return average position of document hits within their evidence set (only including textually matched docs)
def average_position_of_doc_hits_text_match(path):
    with open(path, 'r') as file:
        data = json.load(file)

        doc_hit_positions = []

        for record in data:
            target_docs = []
            for target in record["target"]:
                if target["doc_id"] not in target_docs:
                    target_docs.append(target["doc_id"])

            index = 1
            for evidence in record["evidence"]:
                # if evidence["doc_hit"] and evidence["method"] == "text_match":
                if evidence["doc_id"] in target_docs and evidence["method"] == "text_match":
                    doc_hit_positions.append(index)
                index += 1

        return sum(doc_hit_positions)/len(doc_hit_positions)

def run_FEVERISH_1(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_1.json')
    batch_limit = 200

    run_without_passage(output_path, "FEVERISH_1", claim_db_path, wiki_db_path, batch_limit, seed)

def analyze_FEVERISH_1():
    pass

def run_FEVERISH_3(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3.json')
    batch_limit = 200

    run_without_passage(output_path, "FEVERISH_3", claim_db_path, wiki_db_path, batch_limit, seed)

def analyze_FEVERISH_3(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    avg = execution_avg(json_path)
    fever = FEVER_doc_score(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("FEVER doc score:", (fever * 100), "%")
    print("Execution average:", avg, "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))

def run_FEVERISH_3_1(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3_1.json')
    batch_limit = 200

    run_without_passage(output_path, "FEVERISH_3_1", claim_db_path, wiki_db_path, batch_limit, seed)

def analyze_FEVERISH_3_1(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    avg = execution_avg(json_path)
    fever = FEVER_doc_score(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("FEVER doc score:", (fever * 100), "%")
    print("Execution average:", avg, "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))

def run_feverish_3_2(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3_2.json')
    batch_limit = 200

    run_without_passage(output_path, "FEVERISH_3_2", claim_db_path, wiki_db_path, batch_limit, seed)

def analyze_feverish_3_2(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    avg = execution_avg(json_path)
    fever = FEVER_doc_score(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("FEVER doc score:", (fever * 100), "%")
    print("Execution average:", avg, "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))

def run_feverish_3_3(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3_3.json')
    batch_limit = 200

    run_with_passage(output_path, "FEVERISH_3_3", claim_db_path, wiki_db_path, batch_limit, seed)

def analyze_feverish_3_3(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    pass_recall = passage_recall(json_path)
    pass_precision = passage_precision(json_path)
    comb_recall = combined_recall(json_path)
    comb_precision = combined_precision(json_path)
    avg = execution_avg(json_path)
    fever = FEVER_doc_score(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("Passage recall:", (pass_recall * 100), "%")
    print("Passage precision:", (pass_precision * 100), "%")
    print("Passage F1 score:", f1_score(pass_precision, pass_recall) * 100, "%")
    print("Combined recall:", (comb_recall * 100), "%")
    print("Combined precision:", (comb_precision * 100), "%")
    print("Combined F1 score:", f1_score(comb_precision, comb_recall) * 100, "%")
    print("FEVER doc score:", (fever * 100), "%")
    print("FEVER passage score:", (FEVER_passage_score(json_path) * 100), "%" )
    print("FEVER combined score:", (FEVER_combined_score(json_path) * 100), "%" )
    print("Execution average:", avg, "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))

def run_feverish_3_4(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3_4.json')
    batch_limit = 200

    run_with_passage(output_path, "FEVERISH_3_4", claim_db_path, wiki_db_path, batch_limit, seed, normalize=True)

def analyze_feverish_3_4(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    avg = execution_avg(json_path)
    fever = FEVER_doc_score(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("FEVER doc score:", (fever * 100), "%")
    print("Execution average:", avg, "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))

def run_feverish_3_5(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3_5.json')
    batch_limit = 200

    run_with_passage(output_path, "FEVERISH_3_5", claim_db_path, wiki_db_path, batch_limit, seed, normalize=True)

def analyze_feverish_3_5(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    avg = execution_avg(json_path)
    fever = FEVER_doc_score(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("FEVER doc score:", (fever * 100), "%")
    print("Execution average:", avg, "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))

def run_feverish_3_6(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3_6.json')
    batch_limit = 200

    run_with_passage(output_path, "FEVERISH_3_6", claim_db_path, wiki_db_path, batch_limit, seed, normalize=True, flush_questions=True)

def analyze_feverish_3_6(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    pass_recall = passage_recall(json_path)
    pass_precision = passage_precision(json_path)
    comb_recall = combined_recall(json_path)
    comb_precision = combined_precision(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("Passage recall:", (pass_recall * 100), "%")
    print("Passage precision:", (pass_precision * 100), "%")
    print("Passage F1 score:", f1_score(pass_precision, pass_recall) * 100, "%")
    print("Combined recall:", (comb_recall * 100), "%")
    print("Combined precision:", (comb_precision * 100), "%")
    print("Combined F1 score:", f1_score(comb_precision, comb_recall) * 100, "%")
    print("FEVER doc score:", (FEVER_doc_score(json_path) * 100), "%")
    print("FEVER passage score:", (FEVER_passage_score(json_path) * 100), "%" )
    print("FEVER combined score:", (FEVER_combined_score(json_path) * 100), "%" )
    print("Execution average:", execution_avg(json_path), "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))

def run_feverish_3_7(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'FEVERISH_3_7.json')
    batch_limit = 200

    run_with_passage(output_path, "FEVERISH_3_7", claim_db_path, wiki_db_path, batch_limit, seed, normalize=True, flush_questions=True)

def analyze_feverish_3_7(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    pass_recall = passage_recall(json_path)
    pass_precision = passage_precision(json_path)
    comb_recall = combined_recall(json_path)
    comb_precision = combined_precision(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("Passage recall:", (pass_recall * 100), "%")
    print("Passage precision:", (pass_precision * 100), "%")
    print("Passage F1 score:", f1_score(pass_precision, pass_recall) * 100, "%")
    print("Combined recall:", (comb_recall * 100), "%")
    print("Combined precision:", (comb_precision * 100), "%")
    print("Combined F1 score:", f1_score(comb_precision, comb_recall) * 100, "%")
    print("FEVER doc score:", (FEVER_doc_score(json_path) * 100), "%")
    print("FEVER passage score:", (FEVER_passage_score(json_path) * 100), "%" )
    print("FEVER combined score:", (FEVER_combined_score(json_path) * 100), "%" )
    print("Execution average:", execution_avg(json_path), "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))
    # method_analysis_passages(json_path)
    method_analysis_passages_unique(json_path)

def run_final_version(claim_db_path, wiki_db_path, output_dir, seed):
    output_path = os.path.join(output_dir, 'final_version.json')
    batch_limit = 200

    run_with_passage(output_path, "final_version", claim_db_path, wiki_db_path, batch_limit, seed, normalize=True, flush_questions=True)

def analyse_final_version(json_path):
    document_recall = doc_recall(json_path)
    document_precision = doc_precision(json_path)
    pass_recall = passage_recall(json_path)
    pass_precision = passage_precision(json_path)
    comb_recall = combined_recall(json_path)
    comb_precision = combined_precision(json_path)
    method_analysis(json_path)
    print("Document recall:", (document_recall * 100), "%")
    print("Document precision:", (document_precision * 100), "%")
    print("Document F1 score:", f1_score(document_precision, document_recall) * 100, "%")
    print("Passage recall:", (pass_recall * 100), "%")
    print("Passage precision:", (pass_precision * 100), "%")
    print("Passage F1 score:", f1_score(pass_precision, pass_recall) * 100, "%")
    print("Combined recall:", (comb_recall * 100), "%")
    print("Combined precision:", (comb_precision * 100), "%")
    print("Combined F1 score:", f1_score(comb_precision, comb_recall) * 100, "%")
    print("FEVER doc score:", (FEVER_doc_score(json_path) * 100), "%")
    print("FEVER passage score:", (FEVER_passage_score(json_path) * 100), "%" )
    print("FEVER combined score:", (FEVER_combined_score(json_path) * 100), "%" )
    print("Execution average:", execution_avg(json_path), "s")
    print("Average document length:", average_doc_length(json_path))
    print("Average position of document hits:", average_position_of_doc_hits_text_match(json_path))
    # method_analysis_passages(json_path)
    # method_analysis_passages_unique(json_path)

def find_substring_indices(haystack, needle):
    start_index = haystack.find(needle)
    if start_index != -1:
        end_index = start_index + len(needle) - 1
        return start_index, end_index
    else:
        return None

def subset_checker(set1, set2):
    # Check if set1 is a subset of set2, items are considered equal if they have the same doc_id and an overlapping start and end
    # Input: (doc_id, (start, end)), (doc_id, (start, end))
    for item1 in set1:
        doc_id1 = item1[0]
        start1 = item1[1][0]
        end1 = item1[1][1]
        found = False
        for item2 in set2:
            doc_id2 = item2[0]
            start2 = item2[1][0]
            end2 = item2[1][1]
            if type(start1) != int or type(end1) != int or type(start2) != int or type(end2) != int:
                continue
            if doc_id1 == doc_id2 and ((start1 < end2) and (end1 > start2)):
                found = True
                break
        if not found:
            return False
    return True

def unicode_normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string

def reformat_punct(text):
    # Remove spaces before and after punctuation
    text = re.sub(r'\s([.,!?;:"](?:\s|$))', r'\1', text)
    return text

def remove_tags(line):
    line_parts = line.split('\t')
    if len(line_parts) > 1:
        return line_parts[0]
    return ""

def remove_markup(text):
    # Remove tab characters used to seperate lines with a number in front (e.g. "1\tText")
    text = re.sub(r'\d+\t', '', text)

    # Remove tab characters and newlines
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'\n', ' ', text)
    return text

def normalize_lines(lines):
    return_lines = []

    lines = re.split(re.compile(r'\d+\t'), lines)

    for line in lines:
        normalized_text = unicode_normalize(line)
        if line:
            converted_text = convert_brc(normalized_text)
            converted_text = remove_tags(converted_text)
            converted_text = reformat_punct(converted_text)
            converted_text = remove_markup(converted_text)
            return_lines.append(converted_text)
    return return_lines