import os
import time
from transformers import pipeline, AutoTokenizer, DistilBertForSequenceClassification
from train.relevancy_classification import create_relevancy_dataset, train_relevancy_model, evaluate_relevancy_model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_output_dir = os.path.join(current_dir, '..', 'data', 'train')
    dataset_file = os.path.join(data_output_dir, 'relevancy_classification.json')
    relevancy_model_name = "distilbert/distilbert-base-uncased"

    dataset_creation = input("Do you want to create a new dataset? (y/n): ")
    if dataset_creation == "y":
        db_path = os.path.join(current_dir, '..', 'data', 'data.db')

        num_claims = input("Enter the number of claims to use: ")
        num_claims = int(num_claims) if num_claims else 10000

        create_relevancy_dataset(db_path, data_output_dir, limit=num_claims)

    model_creation = input("Do you want to train a new model? (y/n): ")
    if model_creation == "y":
        model_output_dir = os.path.join(current_dir, '..', 'models', 'relevancy_classification')
        train_relevancy_model(dataset_file, relevancy_model_name, model_output_dir)

    run_model = input("Do you want to run the model? (y/n): ")
    if run_model == "y":
        model_output_dir = os.path.join(current_dir, '..', 'models', 'relevancy_classification')
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

    model_evaluation = input("Do you want to evaluate a model? (y/n): ")
    if model_evaluation == "y":
        evaluate_relevancy_model(dataset_file, model_output_dir)

if __name__ == "__main__":
    main()