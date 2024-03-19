import os
import time
from transformers import pipeline, AutoModel, AutoTokenizer, DistilBertForSequenceClassification
from train.relevancy_classification import create_dataset, train_model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_output_dir = os.path.join(current_dir, '..', 'data', 'train')
    model_output_dir = os.path.join(current_dir, '..', 'models', 'relevancy_classification')

    dataset_creation = input("Do you want to create a new dataset? (y/n): ")
    if dataset_creation == "y":
        db_path = os.path.join(current_dir, '..', 'data', 'data.db')

        num_claims = input("Enter the number of claims to use: ")
        num_claims = int(num_claims) if num_claims else 10000
        create_dataset(db_path, data_output_dir, limit=num_claims)

    model_creation = input("Do you want to train a new model? (y/n): ")
    if model_creation == "y":
        dataset_file = os.path.join(data_output_dir, 'relevancy_classification.json')
        model_name = "distilbert/distilbert-base-uncased"
        train_model(dataset_file, model_name, model_output_dir)

    run_model = input("Do you want to run the model? (y/n): ")
    if run_model == "y":
        model = DistilBertForSequenceClassification.from_pretrained(model_output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
        pipe = pipeline('text-classification', model=model, tokenizer=tokenizer)
        claim = "Murda Beatz's real name is Marshall Mathers."
        sentence = "Murda Beatz's real name is Robert Junior."
        input_pair = f"{claim} [SEP] {sentence}"
        start_time = time.time()
        result = pipe(input_pair)
        print("Inference time:", time.time() - start_time)
        print(result)

if __name__ == "__main__":
    main()