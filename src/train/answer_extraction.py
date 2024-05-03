from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, T5TokenizerFast, T5ForConditionalGeneration, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, TrainerCallback
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
import numpy as np
from tqdm import tqdm
import spacy
from sklearn.model_selection import train_test_split
import json
import numpy as np
import torch

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
        # Remove duplicates
        answers['text'] = list(set(answers['text']))

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

def format_dataset_text2text_sentence(dataset):
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

    nlp = spacy.load("en_core_web_sm")

    for context, answers in tqdm(context_to_answers.items()):
        doc = nlp(context)

        # Split the context into sentences
        for sentence in doc.sents:
            # Append the whole context with the sentence market with a <s> token within the context
            formatted_data['context'].append(context.replace(sentence.text, f"<s> {sentence.text} </s>"))

            # Find out which answers are in this sentence
            sentence_answers = []
            for answer in answers['text']:
                if answer in sentence.text:
                    sentence_answers.append(answer)

            # Remove duplicates
            sentence_answers = list(set(sentence_answers))

            # Append the answers
            formatted_data['answers'].append("||".join(sentence_answers))

    return formatted_data

def train_answer_extraction_model_text2text_sentence(output_dir):
    # Same as above, but each context is split into sentences, the answers from that particular sentence are extracted

    # Initialize the tokenizer and model
    tokenizer = T5TokenizerFast.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Load and process the dataset
    train_dataset = load_dataset("squad_v2", split="train")
    validation_dataset = load_dataset("squad_v2", split="validation")

    # Format the datasets
    train_examples = format_dataset_text2text_sentence(train_dataset)
    validation_examples = format_dataset_text2text_sentence(validation_dataset)
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
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,  
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=2e-5,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=1500,
        save_steps=1500,
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
        eval_dataset=validation_dataset
    )

    # Start training
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def fine_tune_answer_extraction_model_text2text_sentence(output_dir, data_dir):
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("vabatista/t5-small-answer-extraction-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("vabatista/t5-small-answer-extraction-en")

    # tokenizer = T5TokenizerFast.from_pretrained('t5-small')
    # model = T5ForConditionalGeneration.from_pretrained('t5-small')

    with open(data_dir, 'r') as file:
      dataset = json.load(file)["data"]
    
    formatted_data = {
        'inputs': [],
        'targets': [],
    }

    # Format the dataset into the required format (context, answers)
    for item in dataset:
        context = item['text']
        answers = item['entities']
        
        formatted_data['inputs'].append("extract answers: <ha> " + context + " <ha>")
        formatted_data['targets'].append("<sep> ".join(answers))

    dataset = Dataset.from_dict(formatted_data)

    # Split the dataset into training and validation sets
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    validation_dataset = dataset['test']

    # Tokenize the datasets
    tokenize_function = partial(tokenize_text2text, tokenizer=tokenizer)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,  
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        learning_rate=5e-7,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=20,
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
        callbacks=[PrintTextCallback(tokenizer, validation_dataset, device, print_every=10)]
    )

    # Start training
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

class PrintTextCallback(TrainerCallback):
    """A custom callback that prints inputs and outputs from the model periodically."""

    def __init__(self, tokenizer, eval_dataset, device, print_every=100):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.device = device  # Store the device
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        """This function gets called at the end of each training step."""
        if state.global_step % self.print_every == 0:
            # Randomly select an example from the evaluation dataset
            example = np.random.choice(self.eval_dataset)
            inputs = example['inputs']
            targets = example['targets']

            # Tokenize and generate output, ensure tensors are on the right device
            input_ids = self.tokenizer(inputs, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids
            input_ids = input_ids.to(self.device)  # Move input tensor to the correct device

            # Generate outputs, assuming the model is already on the correct device
            outputs = kwargs['model'].generate(input_ids, max_length=512)

            # Decode and print
            decoded_inputs = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            decoded_outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            print(f"\nStep {state.global_step}:")
            print(f"Input: {decoded_inputs}")
            print(f"Predicted Output: {decoded_outputs}")
            print(f"Actual Target: {targets}")