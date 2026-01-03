import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback

PATH_FOR_TRAINING_SET = 'D:/ACE_UCV/Master_Anul_I/NLPTM/train_set4.csv'
OUTPUT_DIR = 'D:/ACE_UCV/Master_Anul_I/NLPTM/Output'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

MODEL_NAME = "facebook/bart-large"


df = pd.read_csv(PATH_FOR_TRAINING_SET, encoding='cp1252')


hf_dataset = Dataset.from_pandas(df)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def preprocess_data(examples):
    """
    Creates training prompts that match the inference format.
    Critical: Must access batch elements with [i] indexing.
    """
    inputs = []
    
    # Iterate through each example in the batch
    for i in range(len(examples['name'])):
        # Build prompt matching your excellent training data pattern
        prompt_text = (
            f"Task: Personalize this exercise for the student. "
            f"Adapt it to their interests and learning style.\n\n"
            f"Student Profile:\n"
            f"Name: {examples['name'][i]}\n"
            f"Age: {examples['age'][i]}\n"
            f"Interests: {examples['interests'][i]}\n"
            f"Learning Style: {examples['learning_style'][i]}\n"
            f"Motivation: {examples['motivation'][i]}\n\n"
            f"Original Exercise: {examples['exercise_normal'][i]}\n\n"
            f"Personalized Exercise:"
        )
        inputs.append(prompt_text)
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )
    
    # Tokenize target labels
    labels = tokenizer(
        examples['exercise_personalized'],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding='max_length'
    )
    
    # Add labels to model inputs
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.gradient_checkpointing_enable()

model.to(device)
print(f"Successfully loaded model on {device}")

# Split dataset into training and validation
split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
print(f"Training samples: {len(split_dataset['train'])}")
print(f"Validation samples: {len(split_dataset['test'])}")


tokenized_dataset = split_dataset.map(
    preprocess_data, 
    batched=True,
    remove_columns=split_dataset['train'].column_names
)
print("Dataset tokenized successfully")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=15,
    per_device_train_batch_size=2,  
    per_device_eval_batch_size=2,  
    gradient_accumulation_steps=2,  
    warmup_steps=200,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=10,
    fp16=True,
    fp16_full_eval=True,  
    evaluation_strategy="steps", 
    eval_steps=50,               
    save_strategy="steps",        
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    gradient_checkpointing=True,
    max_grad_norm=1.0, 
    optim="adafactor",

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(f"{OUTPUT_DIR}/final_bart_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_bart_model")
print(f"Training complete. Model saved to {OUTPUT_DIR}/final_bart_model")