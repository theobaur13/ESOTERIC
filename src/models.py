class Evidence:
    def __init__(self, query, evidence_text, id=None, doc_id=None, doc_score=0, sentences=None, embedding=None, doc_retrieval_method=None):
        self.query = query

        self.id = id
        self.doc_id = doc_id
        self.doc_score = doc_score
        self.evidence_text = evidence_text
        self.sentences = sentences if sentences is not None else []

        self.embedding = embedding
        self.doc_retrieval_method = doc_retrieval_method

    def set_evidence_sentences(self, sentences):
        self.sentences = sentences

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def merge_overlapping_sentences(self):
        self.sentences = sorted(self.sentences, key=lambda x: x.start)
        merged_sentences = []

        for sentence in self.sentences:
            if not merged_sentences:
                merged_sentences.append(Sentence(
                    sentence=sentence.sentence,
                    score=sentence.score,
                    start=sentence.start,
                    end=sentence.end
                ))
            else:
                last = merged_sentences[-1]
                if sentence.start <= last.end:
                    score = max(last.score, sentence.score)
                    start = min(last.start, sentence.start)
                    end = max(last.end, sentence.end)
                    merged_sentence = Sentence(
                        score=score,
                        start=start,
                        end=end,
                        sentence = self.evidence_text[start:end]
                    )
                    merged_sentences[-1] = merged_sentence
                else:
                    merged_sentences.append(Sentence(
                        sentence=sentence.sentence,
                        score=sentence.score,
                        start=sentence.start,
                        end=sentence.end
                    ))
        self.sentences = merged_sentences

    def __str__(self):
        return f"Query: {self.query}\nDoc ID: {self.doc_id}\nDoc Score: {self.doc_score}\nEvidence Text: {self.evidence_text}\nSentences: {self.sentences}\nDoc Retrieval Method: {self.doc_retrieval_method}"

class EvidenceWrapper:
    def __init__(self, query):
        self.query = query
        self.evidences = []

    def add_evidence(self, evidence):
        self.evidences.append(evidence)

    def get_evidences(self):
        return self.evidences
    
    def get_evidence_by_id(self, id):
        for evidence in self.evidences:
            if evidence.id == id:
                return evidence
        return None
    
    def get_claim(self):
        return self.query
    
    def remove_evidence(self, id):
        self.evidences = [evidence for evidence in self.evidences if evidence.id != id]

    def sort_by_doc_score(self):
        self.evidences = sorted(self.evidences, key=lambda x: x.doc_score, reverse=True)

    def sort_by_sentence_score(self):
        for evidence in self.evidences:
            evidence.sentences = sorted(evidence.sentences, key=lambda x: x.score, reverse=True)

        self.evidences = sorted(self.evidences, key=lambda x: x.sentences[0].score if x.sentences else 0, reverse=True)

    def seperate_sort(self):
        # Display documents containing passages first, ordered by their passage score, and then to display documents without passages, ordered by their document score
        evidences_with_sentences = []
        evidences_without_sentences = []
        for evidence in self.evidences:
            if evidence.sentences:
                evidences_with_sentences.append(evidence)
            else:
                evidences_without_sentences.append(evidence)

        evidences_with_sentences.sort(key=lambda x: max(sentence.score for sentence in x.sentences), reverse=True)
        evidences_without_sentences.sort(key=lambda x: x.doc_score, reverse=True)

        self.evidences = evidences_with_sentences + evidences_without_sentences

        def __str__(self):
            return f"Query: {self.query}\nEvidences: {self.evidences}"

class Sentence:
    def __init__(self, sentence=None, score=0, doc_id=None, start=None, end=None, question=None, method=None):
        self.doc_id = doc_id
        self.sentence = sentence
        self.score = score
        self.start = start
        self.end = end
        self.question = question
        self.method = method

    def set_start_end(self, text):
        self.start = text.find(self.sentence)
        self.end = self.start + len(self.sentence)

    def __str__(self):
        return f"Doc ID: {self.doc_id}\nSentence ID: {self.sent_id}\nSentence: {self.sentence}\nScore: {self.score}"