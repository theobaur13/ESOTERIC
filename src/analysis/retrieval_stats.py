def method_performance(data):
    # Calculate the proportion of hits for each method
    methods = ["title_match", "disambiguation", "text_match"]

    method_hits = {}
    for method in methods:
        method_hits[method] = 0

    for claim in data:
        for evidence in claim["evidence"]:
            if evidence["hit"]:
                method_hits[evidence["method"]] += 1

    total_hits = sum(method_hits.values())
    for method in methods:
        print(f"Proportion of hits for {method}: {method_hits[method] / total_hits * 100}%")