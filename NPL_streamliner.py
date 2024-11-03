import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import numpy as np
from deep_translator import GoogleTranslator

# Function definitions
def read_and_clean_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    text = re.sub(r'\n+', ' ', text)  # Remove newline characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = text.strip()  # Remove leading and trailing whitespaces
    text = text.lower()  # Convert to lowercase
    return text

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

def save_txt(text, file_path):
    with open(file_path, "w") as file:
        file.write(text)

def back_translate(text, src_lang='en', mid_lang='de'):
    translator = GoogleTranslator(source=src_lang, target=mid_lang)
    translated = translator.translate(text)
    back_translated = GoogleTranslator(source=mid_lang, target=src_lang).translate(translated)
    return back_translated

# Preprocess text data
file_path = '/Users/user/Desktop/cse485/MyEssay.txt'
text_data = read_and_clean_txt(file_path)
sentences = split_into_sentences(text_data)
train_sentences, val_sentences = train_test_split(sentences, test_size=0.1)

# Augment dataset with back-translation
augmented_sentences = [back_translate(sentence) for sentence in train_sentences]

# Combine original and augmented sentences
all_sentences = train_sentences + augmented_sentences

# Save the augmented dataset
augmented_train_text = " ".join(all_sentences)
save_txt(augmented_train_text, "/Users/user/Desktop/cse485/MyEssay_augmented_train.txt")

# Save validation dataset
val_text = " ".join(val_sentences)
save_txt(val_text, "/Users/user/Desktop/cse485/MyEssay_val.txt")

# Path to the augmented dataset
augmented_train_file_path = "/Users/user/Desktop/cse485/MyEssay_augmented_train.txt"
val_file_path = "/Users/user/Desktop/cse485/MyEssay_val.txt"

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)

def load_data_collator(tokenizer, mlm=False):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)

def train_and_evaluate(train_file_path, val_file_path, model_name, output_dir, num_train_epochs, per_device_train_batch_size, learning_rate, save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    val_dataset = load_dataset(val_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        save_steps=save_steps
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    
    # Evaluate the model
    eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
    
    # Print all metrics to inspect available keys
    print("Train Metrics:", metrics)
    print("Eval Metrics:", eval_metrics)
    
    return eval_metrics.get('eval_loss', float('inf'))  # Return eval_loss if available, else inf

# Grid search setup
hyperparameter_grid = {
    'num_train_epochs': [2, 3],
    'per_device_train_batch_size': [4, 8],
    'learning_rate': [5e-5, 3e-5],
    'save_steps': [100, 200]
}

best_hyperparams = None
best_eval_loss = float('inf')

# # Perform grid search
for num_train_epochs in hyperparameter_grid['num_train_epochs']:
    for per_device_train_batch_size in hyperparameter_grid['per_device_train_batch_size']:
        for learning_rate in hyperparameter_grid['learning_rate']:
            for save_steps in hyperparameter_grid['save_steps']:
                output_dir = f'/Users/user/Desktop/cse485/model_epochs{num_train_epochs}_batch{per_device_train_batch_size}_lr{learning_rate}_steps{save_steps}'
                eval_loss = train_and_evaluate(
                    train_file_path=augmented_train_file_path,
                    val_file_path=val_file_path,
                    model_name='gpt2',
                    output_dir=output_dir,
                    num_train_epochs=num_train_epochs,
                    per_device_train_batch_size=per_device_train_batch_size,
                    learning_rate=learning_rate,
                    save_steps=save_steps,
                )

                print(f'Hyperparameters: epochs={num_train_epochs}, batch_size={per_device_train_batch_size}, learning_rate={learning_rate}, save_steps={save_steps}')
                print(f'Evaluation loss: {eval_loss}\n')

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_hyperparams = {
                        'num_train_epochs': num_train_epochs,
                        'per_device_train_batch_size': per_device_train_batch_size,
                        'learning_rate': learning_rate,
                        'save_steps': save_steps
                    }

print('Best hyperparameters:')
print(best_hyperparams)
print(f'Best evaluation loss: {best_eval_loss}')

# Best hyperparameters from the grid search
best_hyperparams = {
                        'num_train_epochs': 3, #3
                        'per_device_train_batch_size': 4, #4
                        'learning_rate': 5e-05, #5e-05
                        'save_steps': 100 #100
                    }

# Train the Final Model with the Best Hyperparameters
def train_final_model(train_file_path, val_file_path, best_params, model_name='gpt2', output_dir='/Users/user/Desktop/cse485/best_model'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    val_dataset = load_dataset(val_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=best_params['num_train_epochs'],
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        evaluation_strategy="epoch",
        learning_rate=best_params['learning_rate'],
        save_steps=best_params['save_steps'],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)  # Ensure the tokenizer is saved properly

# Train the final model
train_final_model(augmented_train_file_path, val_file_path, best_hyperparams)

# Generate text
def generate_text(model_path, sequence, max_length=50):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
        temperature=0.7,  # Adjusted temperature for more deterministic responses
        num_return_sequences=1,  # Generate only one response
    )
    print(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

# Path to the fine-tuned model
model_path = "/Users/user/Desktop/cse485/best_model"

# Generate text examples with more context
sequence1 = "\nTell me about the history of Rome."
generate_text(model_path, sequence1, max_length=50)

sequence2 = "\nTell me about writings of Shakespeare?"
generate_text(model_path, sequence2, max_length=50)

sequence3 = "Is ASU a good college?"
generate_text(model_path, sequence3, max_length=50)