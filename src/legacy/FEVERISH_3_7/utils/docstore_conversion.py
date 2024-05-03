from haystack import Document
from haystack.document_stores import InMemoryDocumentStore

def wrapper_to_docstore(evidence_wrapper):
    doc_store = InMemoryDocumentStore()
    for evidence in evidence_wrapper.get_evidences():
        id = evidence.id
        doc_id = evidence.doc_id
        content = evidence.evidence_text
        content_type = "text"
        embedding = evidence.embedding
        meta = {"doc_id": doc_id}
        
        # if not already in doc store, add to doc store
        if id not in [d.id for d in doc_store.get_all_documents()]:
            doc = Document(id=id, doc_id=doc_id, content=content, content_type=content_type, embedding=embedding, meta=meta)
            doc_store.write_documents([doc])
    return doc_store

def listdict_to_docstore(listdict):
    doc_store = InMemoryDocumentStore()
    for doc in listdict:
        id = doc['id']
        doc_id = doc['doc_id']
        content = doc['text']
        content_type = "text"
        embedding = doc['embedding']
        meta = {"doc_id": doc_id}
        
        # if not already in doc store, add to doc store
        if id not in [d.id for d in doc_store.get_all_documents()]:
            doc = Document(id=id, doc_id=doc_id, content=content, content_type=content_type, embedding=embedding, meta=meta)
            doc_store.write_documents([doc])
    return doc_store