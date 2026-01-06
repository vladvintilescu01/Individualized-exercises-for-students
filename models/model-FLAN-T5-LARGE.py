import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, EarlyStoppingCallback

# Global variables
PATH_FOR_TRAINING_SET = '../dataset/train_set.csv'
OUTPUT_DIR = 'Output'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Use FLAN-T5-Large - better for instruction following and creative generation
MODEL_NAME = "google/flan-t5-large"

# Load training set 
df = pd.read_csv(PATH_FOR_TRAINING_SET, encoding='cp1252')

# Convert DataFrame to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocess all data 
def preprocess_data(examples):
    """
    Creates training prompts that match the inference format.
    Critical: Must access batch elements with [i] indexing.
    """
    inputs = []
    
    # Iterate through each example in the batch
    for i in range(len(examples['name'])):
        # Build simplified prompt
        prompt_text = (
            f"Task: Personalize this exercise: {examples['exercise_normal'][i]} for the next student: "
            f"Adapt it to their interests: {examples['interests'][i]} and learning style: {examples['learning_style'][i]}.\n\n"
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
    
    # Replace padding token id's of the labels by -100 so it's ignored by the loss
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels["input_ids"]
    ]
    
    # Add labels to model inputs
    model_inputs['labels'] = labels['input_ids']
    
    return model_inputs

# Load model on GPU with memory optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Enable gradient checkpointing to reduce memory usage
model.gradient_checkpointing_enable()

model.to(device)
print(f"Successfully loaded model on {device}")

# Split dataset into training and validation
split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
print(f"Training samples: {len(split_dataset['train'])}")
print(f"Validation samples: {len(split_dataset['test'])}")

# Apply preprocessing function
tokenized_dataset = split_dataset.map(
    preprocess_data, 
    batched=True,
    remove_columns=split_dataset['train'].column_names
)
print("Dataset tokenized successfully")

# Training configuration optimized for 6GB VRAM and better quality
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    fp16=True,
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
    label_smoothing_factor=0.1,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(f"{OUTPUT_DIR}/final_flan_t5_large_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_flan_t5_large_model")
print(f"Training complete. Model saved to {OUTPUT_DIR}/final_flan_t5_large_model")