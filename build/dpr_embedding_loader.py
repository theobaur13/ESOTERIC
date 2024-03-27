from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import DensePassageRetriever
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Connect to Elasticsearch
document_store = ElasticsearchDocumentStore(
    host=os.environ.get("ES_HOST_URL"), port=os.environ.get("ES_PORT"), scheme=os.environ.get("ES_SCHEME"),
    username=os.environ.get("ES_USER"), password=os.environ.get("ES_PASS"),
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
    batch_size=2,
)

# Load the document store
document_store.update_embeddings(retriever=retriever)