import matplotlib.pyplot as plt
from analysis.retrieval_analysis import subset_checker

def plot_performance(data):
    # Plot rolling averages to see where the application tends to
    
    recalls = []
    fever_scores = []
    execution_times = []

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
        execution_times.append(execution_avg)

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

        recalls.append((doc_hits / doc_targets, passage_hits / passage_targets, combined_hits / combined_targets))
        fever_scores.append((FEVER_doc_hits / record_count, FEVER_passage_hits / record_count, FEVER_combined_hits / record_count))

    # Plot a line graph
    # X axis is the record number
    # Y axis is the document, passage, and combined recall percentage, FEVER document, passage, and combined score percentage

    plt.figure(figsize=(10, 5))  
    plt.ylim([0, 1])
    plt.plot(recalls)
    plt.title("Recall")
    plt.xlabel("Record Number")
    plt.ylabel("Recall Percentage")
    plt.legend(["Document", "Passage", "Combined"])
    plt.show()  

    plt.figure(figsize=(10, 5))
    plt.ylim([0, 1])
    plt.plot(fever_scores)
    plt.title("FEVER Score")
    plt.xlabel("Record Number")
    plt.ylabel("FEVER Score Percentage")
    plt.legend(["Document", "Passage", "Combined"])
    plt.show()