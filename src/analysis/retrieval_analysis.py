import os
import sqlite3
import json
from tqdm import tqdm
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
import json
import time

def initialiser(database_path, preloaded_claim=None):
    # Set up the database connection
    conn = sqlite3.connect(os.path.join(database_path, 'data.db'))
    cursor = conn.cursor()

    # Select claims to test from db
    if preloaded_claim:
        cursor.execute('''
            SELECT c.claim, GROUP_CONCAT(cd.doc_id || ':' || cd.sent_id, ';') AS doc_sent_pairs
            FROM claims c
            JOIN claim_docs cd ON c.claim_id = cd.claim_id
            WHERE c.claim_id = ?
            GROUP BY c.claim_id
        ''', (preloaded_claim,))
    else:
        cursor.execute('''
            SELECT c.claim, GROUP_CONCAT(cd.doc_id || ':' || cd.sent_id, ';') AS doc_sent_pairs
            FROM claims c
            JOIN claim_docs cd ON c.claim_id = cd.claim_id
            GROUP BY c.claim_id
            ORDER BY RANDOM()
        ''')

    # Initialise evidence retriever
    title_match_docs_limit = 1000
    text_match_search_db_limit = 1000
    text_match_search_k_limit = 15

    title_match_search_threshold = 0
    text_match_search_threshold = 0
    answerability_threshold = 0.01
    
    evidence_retriever = EvidenceRetriever(database_path, title_match_docs_limit=title_match_docs_limit, text_match_search_db_limit=text_match_search_db_limit, text_match_search_k_limit=text_match_search_k_limit, title_match_search_threshold=title_match_search_threshold, text_match_search_threshold=text_match_search_threshold, answerability_threshold=answerability_threshold)

    return cursor, evidence_retriever

def read_scores(ret_path):
    if not os.path.exists(ret_path):
            with open(ret_path, 'w') as file:
                json.dump([], file)

    with open(ret_path, 'r') as file:
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
            target_combined.append((target["doc_id"], target["sent_id"]))
            if target["doc_id"] in evidence_docs:
                target_passages.append((target["doc_id"], target["sent_id"]))

        # Set counters for targets
        doc_targets += len(target_docs)
        passage_targets += len(target_passages)
        combined_targets += len(target_combined)

        evidence_passages = []
        for evidence in record["evidence"]:
            for sent_id in evidence["sent_id"]:
                evidence_passages.append((evidence["doc_id"], str(sent_id)))

            doc_hit = False
            passage_hit = False
            combined_hit = False

            if evidence["doc_id"] in target_docs:
                doc_hit = True
            
            for sent_id in evidence["sent_id"]:
                if (evidence["doc_id"], str(sent_id)) in target_passages:
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
                combined_misses += 1

        # if target_docs are all in evidence_docs, then it's a FEVER doc hit
        if set(target_docs).issubset(evidence_docs):
            FEVER_doc_hits += 1

        # if target_passages are all in evidence_passages, then it's a FEVER passage hit
        if set(target_passages).issubset(evidence_passages):
            FEVER_passage_hits += 1

        if set(target_combined).issubset(evidence_passages):
            FEVER_combined_hits += 1

    return doc_hits, doc_misses, doc_targets, passage_hits, passage_misses, passage_targets, combined_hits, combined_misses, combined_targets, FEVER_doc_hits, FEVER_passage_hits, FEVER_combined_hits, execution_avg, record_count

def skeleton(database_path, output_dir, preloaded_claim=None):
    # Set output path for results
    output_path = os.path.join(output_dir, "retrieval_results.json")

    # Set up the database connection and evidence retriever
    if preloaded_claim:
        cursor, evidence_retriever = initialiser(database_path, preloaded_claim)
    else:
        cursor, evidence_retriever = initialiser(database_path)

    # Set up counters for calculation
    doc_hits, doc_misses, doc_targets, passage_hits, passage_misses, passage_targets, combined_hits, combined_misses, combined_targets, FEVER_doc_hits, FEVER_passage_hits, FEVER_combined_hits, execution_avg, record_count = read_scores(output_path)

    # Iterate through each claim in the test retrieval table randomly
    for row in tqdm(cursor.fetchall()):
        record_count += 1
        claim = row[0]
        evidence_pairs = row[1].split(';')
        
        target_docs = list(set([pair.split(':')[0] for pair in evidence_pairs]))
        print("\nTarget documents:", target_docs)

        start = time.time()
        evidence_wrapper = evidence_retriever.retrieve_evidence(claim)
        end = time.time()
        execution_time = end - start
        print("Execution time:", execution_time)

        target_passages = []
        target_combined = []
        evidence_docs = list(set([evidence.doc_id for evidence in evidence_wrapper.get_evidences()]))

        for pair in evidence_pairs:
            doc_id = pair.split(':')[0]
            sent_id = pair.split(':')[1]
            target_combined.append((doc_id, sent_id))
            if doc_id in evidence_docs:
                target_passages.append((doc_id, sent_id))

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
            result["target"].append({
                "doc_id": doc_id,
                "sent_id": sent_id
            })

        # Check if any documents in the evidence wrapper match the target document
        for evidence in evidence_wrapper.get_evidences():
            doc_hit = False
            sentence_hit = False
            combined_hit = False

            if evidence.doc_id in target_docs:
                doc_hit = True

            for sentence in evidence.sentences:
                if (evidence.doc_id, str(sentence.sent_id)) in target_passages:
                    sentence_hit = True

                if (evidence.doc_id, str(sentence.sent_id)) in target_combined:
                    combined_hit = True

            # Add evidence to the result dictionary
            result["evidence"].append({
                "doc_id": evidence.doc_id,
                "sent_id": [sentence.sent_id for sentence in evidence.sentences],
                # "sent_id": evidence.sent_id,
                "score": str(evidence.doc_score),
                "method" : evidence.doc_retrieval_method,
                "doc_hit" : doc_hit,
                "sentence_hit" : sentence_hit
            })

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
                evidence_passages.append((evidence.doc_id, str(sentence.sent_id)))

        # if target_docs are all in evidence_docs, then it's a FEVER doc hit
        if set(target_docs).issubset(evidence_docs):
            FEVER_doc_hits += 1

        # if target_passages are all in evidence_passages, then it's a FEVER passage hit
        if set(target_passages).issubset(evidence_passages):
            FEVER_passage_hits += 1

        # if target_combined are all in evidence_passages, then it's a FEVER combined hit
        if set(target_combined).issubset(evidence_passages):
            FEVER_combined_hits += 1

        # write line to file
        if not os.path.exists(output_path):
            with open(output_path, 'w') as file:
                json.dump([], file) 

        with open(output_path, 'r') as file:
            data = json.load(file)

        data.append(result)
    
        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

        # print scores
        precision_doc = doc_hits / (doc_hits + doc_misses)
        recall_doc = doc_hits / doc_targets
        precision_passage = passage_hits / (passage_hits + passage_misses)
        recall_passage = passage_hits / passage_targets
        precision_combined = combined_hits / (combined_hits + combined_misses)
        recall_combined = combined_hits / combined_targets
        FEVER_doc_score = FEVER_doc_hits / len(data)
        FEVER_passage_score = FEVER_passage_hits / len(data)
        FEVER_combined_score = FEVER_combined_hits / len(data)
        execution_avg = (execution_avg * record_count + execution_time) / (record_count + 1)

        print("Precision (doc): " + str(precision_doc*100) + "%")
        print("Recall (doc): " + str(recall_doc*100) + "%")
        print("Precision (passage): " + str(precision_passage*100) + "%")
        print("Recall (passage): " + str(recall_passage*100) + "%")
        print("Precision (combined): " + str(precision_combined*100) + "%")
        print("Recall (combined): " + str(recall_combined*100) + "%")
        print("FEVER doc score: " + str(FEVER_doc_score*100) + "%")
        print("FEVER passage score: " + str(FEVER_passage_score*100) + "%")
        print("FEVER combined score: " + str(FEVER_combined_score*100) + "%")
        print("Execution average: " + str(execution_avg) + "s")