import os
from transformers import pipeline
from train.relevancy_classification import create_dataset, train_model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, '..', 'data', 'train')
    print(output_dir)
    db_path = os.path.join(current_dir, '..', 'data', 'data.db')
    # create_dataset(db_path, output_dir)
    dataset_file = os.path.join(output_dir, 'relevancy_classification.json')
    model_name = "FacebookAI/roberta-base"
    model, tokenizer = train_model(dataset_file, model_name, output_dir)
    pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer)
    claim = "Chris Hemsworth appeared in A Perfect Getaway."
    sentence = "Hemsworth has also appeared in the science fiction action film Star Trek (2009)."
    input_pair = f"{claim} [SEP] {sentence}"
    result = pipeline(input_pair)
    print(result)

if __name__ == "__main__":
    main()