import os
import sqlite3
import json
from tqdm import tqdm
from evidence_retrieval.evidence_retrieval import EvidenceRetriever
import json
import time
import unicodedata
import re

### TODO: Migrate the following functions to a separate file and import them here
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

def find_substring_indices(haystack, needle):
    start_index = haystack.find(needle)
    if start_index != -1:
        end_index = start_index + len(needle) - 1
        return start_index, end_index
    else:
        return None

### End of functions to migrate

def subset_checker(set1, set2):
    # Check if set1 is a subset of set2, items are considered equal if they have the same doc_id and an overlapping start and end
    for item1 in set1:
        doc_id1 = item1[0]
        start1 = item1[1][0]
        end1 = item1[1][1]
        found = False
        for item2 in set2:
            doc_id2 = item2[0]
            start2 = item2[1][0]
            end2 = item2[1][1]
            if doc_id1 == doc_id2 and ((start1 < end2) and (end1 > start2)):
                found = True
                break
        if not found:
            return False
    return True

def print_scores(doc_hits, doc_misses, doc_targets, passage_hits, passage_misses, passage_targets, combined_hits, combined_misses, combined_targets, FEVER_doc_hits, FEVER_passage_hits, FEVER_combined_hits, execution_avg, record_count):
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
        batch_limit = int(input("Enter index of maximum wiki-pages to load (default is all): "))
        difficult_subset = input("Would you like to test the difficult subset of claims? (y/n)\n")
        if difficult_subset == "y":
            cursor.execute(''' 
                SELECT c.claim, GROUP_CONCAT(cd.doc_id || ':' || cd.sent_id, ';') AS doc_sent_pairs
                FROM claims c
                JOIN claim_docs cd ON c.claim_id = cd.claim_id
                WHERE c.claim_id IN (
                    SELECT cd2.claim_id
                    FROM claim_docs cd2
                    WHERE cd2.doc_no BETWEEN 1 AND ?
                    GROUP BY cd2.claim_id
                    HAVING COUNT(DISTINCT cd2.doc_id) > 1
                )
                AND cd.doc_no BETWEEN 1 AND ?
                GROUP BY c.claim_id
                ORDER BY RANDOM()
            ''' , (batch_limit, batch_limit))
        elif difficult_subset == "n":
            cursor.execute('''
                SELECT c.claim, GROUP_CONCAT(cd.doc_id || ':' || cd.sent_id, ';') AS doc_sent_pairs
                FROM claims c
                JOIN claim_docs cd ON c.claim_id = cd.claim_id
                                AND cd.doc_no BETWEEN 1 AND ?
                GROUP BY c.claim_id
                ORDER BY RANDOM()
            ''', (batch_limit,))

    # Initialise evidence retriever
    title_match_docs_limit = 1000
    text_match_search_db_limit = 1000

    title_match_search_threshold = 0
    answerability_threshold = 0.65
    reader_threshold = 0.7
    
    evidence_retriever = EvidenceRetriever(title_match_docs_limit=title_match_docs_limit, text_match_search_db_limit=text_match_search_db_limit, title_match_search_threshold=title_match_search_threshold, answerability_threshold=answerability_threshold, reader_threshold=reader_threshold)

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
                combined_misses += 1

        # if target_docs are all in evidence_docs, then it's a FEVER doc hit
        if set(target_docs).issubset(evidence_docs):
            FEVER_doc_hits += 1

        # if target_passages are all in evidence_passages, then it's a FEVER passage hit
        if subset_checker(target_passages, evidence_passages):
            FEVER_passage_hits += 1

        if subset_checker(target_combined, evidence_passages):
            FEVER_combined_hits += 1

    print_scores(doc_hits, doc_misses, doc_targets, passage_hits, passage_misses, passage_targets, combined_hits, combined_misses, combined_targets, FEVER_doc_hits, FEVER_passage_hits, FEVER_combined_hits, execution_avg, record_count)

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

        # Run the evidence retrieval
        start = time.time()
        evidence_retriever.flush_questions()
        evidence_wrapper = evidence_retriever.retrieve_evidence(claim)
        end = time.time()
        execution_time = end - start
        print("Execution time:", execution_time)

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
        for doc_id in target_docs:
            # get text from documents db for doc_id
            cursor.execute('''
                SELECT text
                FROM documents
                WHERE doc_id = ?
            ''', (doc_id,))

            lines = cursor.fetchone()[0]

            text = " ".join(normalize_lines(lines))
            split_lines = normalize_lines(lines)

            combined_relevant_lines = []
            passage_relevant_lines = []
            for index, line in enumerate(split_lines):
                if (doc_id, str(index)) in target_combined:
                    combined_relevant_lines.append(line)
                    if (doc_id, str(index)) in target_passages:
                        passage_relevant_lines.append(line)

            for line in combined_relevant_lines:
                indices = find_substring_indices(text, line)
                if indices:
                    target_span_positions_combined.append({"doc_id": doc_id, "start": indices[0], "end": indices[1]})

            for line in passage_relevant_lines:
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

            for span in target_span_positions_combined:
                if span["doc_id"] == doc_id:
                    span_start = span["start"]
                    span_end = span["end"]
                    break

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

        # write line to file
        if not os.path.exists(output_path):
            with open(output_path, 'w') as file:
                json.dump([], file) 

        with open(output_path, 'r') as file:
            data = json.load(file)

        data.append(result)
    
        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

        print_scores(doc_hits, doc_misses, doc_targets, passage_hits, passage_misses, passage_targets, combined_hits, combined_misses, combined_targets, FEVER_doc_hits, FEVER_passage_hits, FEVER_combined_hits, execution_avg, record_count)