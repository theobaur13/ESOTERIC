from models import Evidence, EvidenceWrapper
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class EvidenceRetriever:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self.load_data(dataset_path)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def load_data(self, dataset_path):
        if dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path, orient='records')
        elif dataset_path.endswith('.jsonl'):
            df = pd.read_json(dataset_path, lines=True)
        else:
            raise ValueError('Unsupported file format')
        return df

    def retrieve_evidence(self, claim):
        #evidence_wrapper = self.use_cosine_similarity(claim)
        evidence_wrapper = self.use_semantic_search(claim)
        return evidence_wrapper
    
    def use_cosine_similarity(self, claim):
        claim_embedding = self.model.encode(claim.text, convert_to_tensor=True)
        evidence_embeddings = self.model.encode(self.data['justification'].tolist()[:100], convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(claim_embedding, evidence_embeddings)

        df = pd.DataFrame({'id': self.data['id'].tolist()[:100], 'sentence': self.data['justification'].tolist()[:100], 'score': cos_scores[0]})
        df = df.sort_values(by='score', ascending=False)

        evidence_wrapper = EvidenceWrapper()
        for index, row in df.iterrows():
            if row['score'] > 0.5:
                evidence_wrapper.add_evidence(Evidence(claim.text, row['sentence'], row['score'], row['id']))

        return evidence_wrapper
    
    def use_semantic_search(self, claim):
        query_chunk_size = 100
        corpus_chunk_size = 10000
        top_k = 10

        query_embedding = self.model.encode(claim.text, convert_to_tensor=True)
        query_embedding = query_embedding.to('cpu')

        corpus_embeddings = self.model.encode(self.data['justification'].tolist()[:100], convert_to_tensor=True)
        corpus_embeddings = corpus_embeddings.to('cpu')

        search_result = util.semantic_search(query_embedding, corpus_embeddings, query_chunk_size=query_chunk_size, corpus_chunk_size=corpus_chunk_size, top_k=top_k)

        evidence_wrapper = EvidenceWrapper()
        for result in search_result[0]:
            claim_text = claim.text
            evidence = self.data['justification'].tolist()[result['corpus_id']]
            score = result['score']
            evidence_id = self.data['id'].tolist()[result['corpus_id']]
            
            if score > 0.5:
                evidence_wrapper.add_evidence(Evidence(claim_text, evidence, score, evidence_id))

        return evidence_wrapper