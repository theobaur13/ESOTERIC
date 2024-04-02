import os
import argparse
import json
import unicodedata
import re
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import BulkIndexError
from dotenv import load_dotenv

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

def normalize_text(text):
    normalized_text = unicode_normalize(text)
    converted_text = convert_brc(normalized_text)
    converted_text = reformat_punct(converted_text)
    return converted_text

### End of functions to migrate

def main(batch_limit=None):
    load_dotenv(override=True)
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
                    pure_text = data["text"]
                    formatted_text = normalize_text(pure_text)
                    action = {
                        "_index": "documents",
                        "_source": {
                            "doc_id": data['id'],
                            "content": formatted_text,
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