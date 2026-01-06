from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = "../models/Output/final_bart_model"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()  # Set to evaluation mode
print(f"Model loaded successfully on {device}")

test_example = {
    "name": "Dragos",
    "age": 7,
    "interests": "football",
    "learning_style": "visual",
    "motivation": "low",
    "exercise_normal": "A student has 48 apples and wants to put them equally into 6 baskets. How many apples will each basket get?"
}


prompt = (
    f"Task: Personalize this exercise for the student. "
    f"Adapt it to their interests and learning style.\n\n"
    f"Student Profile:\n"
    f"Name: {test_example['name']}\n"
    f"Age: {test_example['age']}\n"
    f"Interests: {test_example['interests']}\n"
    f"Learning Style: {test_example['learning_style']}\n"
    f"Motivation: {test_example['motivation']}\n\n"
    f"Original Exercise: {test_example['exercise_normal']}\n\n"
    f"Personalized Exercise:"
)

print("INPUT PROMPT:")

print(prompt)



inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=128,         
        min_length=30,         
        num_beams=4,             
        early_stopping=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        length_penalty=0.8,      

    )

personalized_exercise = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(personalized_exercise)

