from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets
from functools import partial
from sklearn.model_selection import train_test_split
import numpy as np

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
def train_answer_extraction_model(model_output_dir):
    # Initialize the tokenizer
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
    
    # # Print the first X examples
    # for i in range(2):
    #     # Print the context
    #     print(f"Context: {aggregated_examples[i]['context']}")
    #     # Print the answers
    #     print(f"Answers: {aggregated_examples[i]['answers']}")
    #     # Print the labels
    #     print(f"Labels: {processed_dataset[i]['labels']}")
    #     # Print the words in the context that have a label of 1
    #     for j, label in enumerate(processed_dataset[i]['labels'][0]):
    #         if label == 1:
    #             print(tokenizer.convert_ids_to_tokens(processed_dataset[i]['input_ids'][0][j]))

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