from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, T5TokenizerFast, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
import numpy as np
from tqdm import tqdm

def process_dataset(tokenizer, examples):
    tokenized_inputs = tokenizer(examples['context'], padding='max_length', truncation=True, return_offsets_mapping=True, max_length=512)
    labels = []

    for i, offset_mapping in enumerate(tokenized_inputs['offset_mapping']):
        doc_labels = [-100] * len(offset_mapping)
        answers = examples['answers'][i]

        for answer_start, answer_text in zip(answers['answer_start'], answers['text']):
            answer_end = answer_start + len(answer_text) - 1

            for j, (start, end) in enumerate(offset_mapping):
                if start >= answer_start and end <= answer_end + 1:
                    doc_labels[j] = 1
                elif start == 0 and end == 0:
                    doc_labels[j] = -100
                elif doc_labels[j] != 1:
                    doc_labels[j] = 0

        labels.append(doc_labels)

    tokenized_inputs['labels'] = labels
    tokenized_inputs.pop('offset_mapping')
    return tokenized_inputs

# Main training function
def train_answer_extraction_model_distilBERT(model_output_dir):
    # Initialize the tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Load and process the dataset
    train_dataset = load_dataset("squad_v2", split="train")
    validation_dataset = load_dataset("squad_v2", split="validation")
    dataset = concatenate_datasets([train_dataset, validation_dataset])

    context_to_answers = {}
    for item in dataset:
        context = item['context']
        if context not in context_to_answers:
            context_to_answers[context] = {'text': [], 'answer_start': []}
        context_to_answers[context]['text'].extend(item['answers']['text'])
        context_to_answers[context]['answer_start'].extend(item['answers']['answer_start'])

    aggregated_examples = [{
        'context': context,
        'answers': answers
    } for context, answers in context_to_answers.items()]
    
    process_function = partial(process_dataset, tokenizer)

    # Tokenize and prepare data for training
    processed_dataset = [process_function({'context': [ex['context']], 'answers': [ex['answers']]}) for ex in aggregated_examples]

    processed_data = {
        'input_ids': [example['input_ids'] for example in processed_dataset],
        'attention_mask': [example['attention_mask'] for example in processed_dataset],
        'labels': [example['labels'] for example in processed_dataset]
    }

    processed_dataset = Dataset.from_dict(processed_data)

    # Split the dataset into training and validation sets
    processed_dataset = processed_dataset.train_test_split(test_size=0.1)
    train_dataset = processed_dataset['train']
    validation_dataset = processed_dataset['test']

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    print(f"Input IDs shape: {np.shape(processed_data['input_ids'])}")
    print(f"Attention Mask shape: {np.shape(processed_data['attention_mask'])}")
    print(f"Labels shape: {np.shape(processed_data['labels'])}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,  
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=2e-5,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=500,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )
    
    # Start training
    trainer.train()
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

def format_dataset_text2text(dataset):
    context_to_answers = {}

    for item in tqdm(dataset):
        context = item['context']
        if context not in context_to_answers:
            context_to_answers[context] = {'text': []}
        context_to_answers[context]['text'].extend(item['answers']['text'])

    formatted_data = {
        'context': [],
        'answers': [],
    }

    for context, answers in context_to_answers.items():
        formatted_data['context'].append(context)
        formatted_data['answers'].append("||".join(answers['text']))

    return formatted_data

def preprocess_text2text(examples):
    inputs = [f"extract answers: {context}" for context in examples["context"]]
    targets = examples["answers"]
    return {"inputs": inputs, "targets": targets}

def tokenize_text2text(examples, tokenizer):
    model_inputs = tokenizer(examples["inputs"], max_length=512, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(examples["targets"], max_length=512, padding="max_length", truncation=True)

    model_inputs["labels"] = targets["input_ids"]
    return model_inputs

def train_answer_extraction_model_text2text(output_dir):
    # Initialize the tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Load and process the dataset
    train_dataset = load_dataset("squad_v2", split="train")
    validation_dataset = load_dataset("squad_v2", split="validation")

    # Format the datasets
    train_examples = format_dataset_text2text(train_dataset)
    validation_examples = format_dataset_text2text(validation_dataset)
    train_dataset = Dataset.from_dict(train_examples)
    validation_dataset = Dataset.from_dict(validation_examples)
    
    # Preprocess the datasets
    train_dataset = train_dataset.map(preprocess_text2text, batched=True)
    validation_dataset = validation_dataset.map(preprocess_text2text, batched=True)

    tokenize_function = partial(tokenize_text2text, tokenizer=tokenizer)

    # Tokenize and prepare data for training
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,  
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=2e-5,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=5,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    # Start training
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)