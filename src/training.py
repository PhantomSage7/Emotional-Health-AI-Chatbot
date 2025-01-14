import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from utils import load_data, preprocess_data

def train_model(train_data):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    # Convert the DataFrame to a Dataset
    dataset = Dataset.from_pandas(train_data)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir='./models/fine_tuned',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=500,      # Log every 500 steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    # Load and preprocess the training data
    train_data = load_data('data/processed/train_data.csv')
    train_data = preprocess_data(train_data)  # Optional preprocessing step
    train_model(train_data)