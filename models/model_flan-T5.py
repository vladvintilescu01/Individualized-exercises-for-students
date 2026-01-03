import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq
)

# Clear GPU memory first
torch.cuda.empty_cache()

# Global variables
PATH_FOR_TRAINING_SET = 'D:/ACE_UCV/Master_Anul_I/NLPTM/train_set4.csv'
OUTPUT_DIR = 'D:/ACE_UCV/Master_Anul_I/NLPTM/Output'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Use FLAN-T5-Base
MODEL_NAME = "google/flan-t5-base"

# Load training set 
df = pd.read_csv(PATH_FOR_TRAINING_SET, encoding='cp1252')

# Convert DataFrame to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Preprocess all data - SIMPLIFIED VERSION
def preprocess_data(examples):
    """
    Creates training prompts that match the inference format.
    """
    inputs = []
    targets = []
    
    # Iterate through each example in the batch
    for i in range(len(examples['name'])):
        # Build simplified prompt
        prompt_text = (
            f"Task: Personalize this exercise: {examples['exercise_normal'][i]} for the next student: "
            f"Adapt it to their interests: {examples['interests'][i]} and learning style: {examples['learning_style'][i]}.\n\n"
        )
        inputs.append(prompt_text)
        targets.append(examples['exercise_personalized'][i])
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    )
    
    # Tokenize targets
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )
    
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Load model on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)
print(f"Successfully loaded model on {device}")

# Split dataset
split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
print(f"Training samples: {len(split_dataset['train'])}")
print(f"Validation samples: {len(split_dataset['test'])}")

# Apply preprocessing
tokenized_dataset = split_dataset.map(
    preprocess_data, 
    batched=True,
    remove_columns=split_dataset['train'].column_names
)
print("Dataset tokenized successfully")

# Create data collator - THIS HANDLES PADDING AND LABELS PROPERLY
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=150,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=10,
    fp16=True,
    eval_strategy="steps",
    eval_steps=50,               
    save_strategy="steps",        
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    label_smoothing_factor=0.1,
    report_to="none",  # Disable wandb/tensorboard
)

# Initialize Trainer with data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    processing_class=tokenizer,
    data_collator=data_collator,  # THIS IS THE KEY!
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

print("Starting training...")
trainer.train()

# Save final model
trainer.save_model(f"{OUTPUT_DIR}/final_flan_t5_base_creative")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_flan_t5_base_creative")
print(f"Training complete. Model saved to {OUTPUT_DIR}/final_flan_t5_base_creative")