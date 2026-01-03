from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Clear GPU memory first
torch.cuda.empty_cache()

# Path to your trained model
MODEL_PATH = "D:/ACE_UCV/Master_Anul_I/NLPTM/Output/final_flan_t5_base_creative"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print(f"Model loaded successfully on {device}")

# Test example
test_example = {
    "interests": "science, technology",
    "learning_style": "visual",
    "exercise_normal": "A student has 48 apples and wants to put them equally into 6 baskets. How many apples will each basket get?"
}

# Create prompt MATCHING the training format
prompt = (
    f"Task: Personalize this exercise: {test_example['exercise_normal']} for the next student: "
    f"Adapt it to their interests: {test_example['interests']} and learning style: {test_example['learning_style']}.\n\n"
)

print("\n" + "="*80)
print("INPUT PROMPT:")
print("="*80)
print(prompt)
print("="*80 + "\n")

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=256,
        min_length=40,
        do_sample=True,
        temperature=0.8,  # Higher for more creativity
        top_p=0.92,
        repetition_penalty=1.3,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

# Decode result
personalized_exercise = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("="*80)
print("PERSONALIZED EXERCISE:")
print("="*80)
print(personalized_exercise)
print("="*80)