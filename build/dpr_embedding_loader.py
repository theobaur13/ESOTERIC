from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(
    host="localhost", port=9200,
    username="elastic", password="password",
    index="documents",
    embedding_field="embedding",
    embedding_dim=768,
)

# Load Dense Passage Retriever
retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False,
    embed_title=True,
)

# Load the document store
document_store.update_embeddings(retriever=retriever)