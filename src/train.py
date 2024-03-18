import os
import time
from transformers import pipeline, AutoModel, AutoTokenizer, DistilBertForSequenceClassification
from train.relevancy_classification import create_dataset, train_model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_output_dir = os.path.join(current_dir, '..', 'data', 'train')
    db_path = os.path.join(current_dir, '..', 'data', 'data.db')
    create_dataset(db_path, data_output_dir)

    dataset_file = os.path.join(data_output_dir, 'relevancy_classification.json')
    model_output_dir = os.path.join(current_dir, '..', 'models', 'relevancy_classification')
    model_name = "distilbert/distilbert-base-uncased"
    train_model(dataset_file, model_name, model_output_dir)

    model = DistilBertForSequenceClassification.from_pretrained(model_output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_output_dir)
    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer)
    claim = "Murda Beatz's real name is Marshall Mathers."
    sentence = "Hemsworth has also appeared in the science fiction action film A Perfect Getaway."
    input_pair = f"{claim} [SEP] {sentence}"
    start_time = time.time()
    result = pipe(input_pair)
    print("Inference time:", time.time() - start_time)
    print(result)

if __name__ == "__main__":
    main()