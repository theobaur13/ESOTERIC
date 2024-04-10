import os
import time
from transformers import pipeline, AutoModel, AutoTokenizer, DistilBertForSequenceClassification, DistilBertTokenizerFast, DistilBertForTokenClassification, T5TokenizerFast, T5ForConditionalGeneration
from train.relevancy_classification import create_relevancy_dataset, train_relevancy_model, evaluate_relevancy_model
from train.span_extraction import create_span_dataset, train_span_model
from train.question_generation import create_question_dataset, train_question_model
from train.answer_extraction import train_answer_extraction_model_distilBERT, train_answer_extraction_model_text2text

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_output_dir = os.path.join(current_dir, '..', 'data', 'train')
    dataset_file = os.path.join(data_output_dir, 'relevancy_classification.json')
    relevancy_model_name = "distilbert/distilbert-base-uncased"
    span_model_name = "SpanBERT/spanbert-base-cased"

    dataset_creation = input("Do you want to create a new dataset? (y/n): ")
    if dataset_creation == "y":
        db_path = os.path.join(current_dir, '..', 'data', 'data.db')

        num_claims = input("Enter the number of claims to use: ")
        num_claims = int(num_claims) if num_claims else 10000

        train_type = input("Do you want to create a dataset for: \nRelevancy classification (r)\nSpan extraction (s)\nQuestion generation (q)\n")
        if train_type == "r":
            create_relevancy_dataset(db_path, data_output_dir, limit=num_claims)
        elif train_type == "s":
            create_span_dataset(db_path, data_output_dir, limit=num_claims)
        elif train_type == "q":
            create_question_dataset(db_path, data_output_dir, limit=num_claims)

    model_creation = input("Do you want to train a new model? (y/n): ")
    if model_creation == "y":
        train_type = input("Do you want to train a: \nRelevancy classification model (r)\nSpan extraction model (s)\nQuestion generation model (q)\nAnswer extraction model (a)\n")
        if train_type == "r":
            model_output_dir = os.path.join(current_dir, '..', 'models', 'relevancy_classification')
            train_relevancy_model(dataset_file, relevancy_model_name, model_output_dir)
        elif train_type == "s":
            model_output_dir = os.path.join(current_dir, '..', 'models', 'span_extraction')
            train_span_model(dataset_file, relevancy_model_name, model_output_dir)
        elif train_type == "q":
            model_output_dir = os.path.join(current_dir, '..', 'models', 'question_generation')
            train_question_model(dataset_file, relevancy_model_name, model_output_dir)
        elif train_type == "a":
            model_output_dir = os.path.join(current_dir, '..', 'models', 'answer_extraction')
            train_answer_extraction_model_text2text(model_output_dir)

    run_model = input("Do you want to run the model? (y/n): ")
    if run_model == "y":
        train_type = input("Do you want to run a: \nRelevancy classification model (r)\nSpan extraction model (s)\nQuestion generation model (q)\nAnswer extraction model (a)\n")
        if train_type == "r":
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
        elif train_type == "a":
            extractor_model_type = input("Enter the type of model to use: \nDistilBERT (d)\nText2Text (t)\n")
            if extractor_model_type == "d":
                model_output_dir = os.path.join(current_dir, '..', 'models', 'answer_extraction')
                model = DistilBertForTokenClassification.from_pretrained(model_output_dir)
                tokenizer = DistilBertTokenizerFast.from_pretrained(model_output_dir)
                pipe = pipeline('token-classification', model=model, tokenizer=tokenizer)
                passage = "Telemundo is a English-language television network."
                start_time = time.time()
                results = pipe(passage)
                print("Inference time:", time.time() - start_time)
                for result in results:
                    if result['entity'] == 'LABEL_1':
                        print(result['word'], result['score'])
            elif extractor_model_type == "t":
                model_output_dir = os.path.join(current_dir, '..', 'models', 'answer_extraction_t2t')
                model = T5ForConditionalGeneration.from_pretrained(model_output_dir)
                tokenizer = T5TokenizerFast.from_pretrained(model_output_dir)
                pipe = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
                passage = "Theodor Baur is German."
                start_time = time.time()
                results = pipe("extract answers: " + passage)
                print("Inference time:", time.time() - start_time)
                print(results)

    model_evaluation = input("Do you want to evaluate a model? (y/n): ")
    if model_evaluation == "y":
        train_type = input("Do you want to evaluate a relevancy classification or span extraction model? (r/s): ")
        if train_type == "r":
            evaluate_relevancy_model(dataset_file, model_output_dir)
        elif train_type == "s":
            pass

if __name__ == "__main__":
    main()