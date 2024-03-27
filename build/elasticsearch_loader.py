import os
import argparse
import json
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError
from dotenv import load_dotenv

def main(batch_limit=None):
    load_dotenv()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'data', 'wiki-pages')

    # Initialize the Elasticsearch document store
    print("Initializing Elasticsearch document store")
    document_store = Elasticsearch(hosts=[os.environ.get("ES_HOST_URL")], basic_auth=(os.environ.get("ES_USER"), os.environ.get("ES_PASS")))

    custom_mapping = {
        "settings": {
            "analysis": {
                "normalizer": {
                    "lowercase_normalizer": {
                        "type": "custom",
                        "filter": ["lowercase"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {
                    "type": "keyword",
                    "normalizer": "lowercase_normalizer"
                },
                "content": {
                    "type": "text"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768
                }
            }
        }
    }

    # Drop any existing documents index
    print("Dropping existing documents index")
    if document_store.indices.exists(index="documents"):
        document_store.indices.delete(index="documents")

    # Recreate the index after deletion to ensure it's clean
    print("Creating new documents index")
    document_store.indices.create(index="documents", body=custom_mapping)

    # Select a subset of files to process
    file_list = os.listdir(dataset_path)
    if batch_limit:
        file_list = file_list[:batch_limit]
    print("Loading " + str(len(file_list)) + " documents into database")

    for file_name in tqdm(file_list):
        file_path = os.path.join(dataset_path, file_name)
        if file_path.endswith('.jsonl'):
            with open(file_path, "r") as file:
                actions = []
                for json_line in file:
                    data = json.loads(json_line)
                    action = {
                        "_index": "documents",
                        "_source": {
                            "doc_id": data['id'],
                            "content": data['lines'],
                            "content_type": "text"
                        }
                    }
                    actions.append(action)

                try:
                    from elasticsearch import helpers
                    helpers.bulk(document_store, actions)
                except BulkIndexError as e:
                    tqdm.write(f"BulkIndexError processing batch in file {file_name}: {e}")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load wiki-pages into database.')
    parser.add_argument('--batch_limit', type=int, help='Limit the number of files processed. Leave blank to process all files.', nargs='?', const=None)
    args = parser.parse_args()
    
    main(batch_limit=args.batch_limit)