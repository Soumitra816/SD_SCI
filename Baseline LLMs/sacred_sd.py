from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
import pandas as pd

# Load the model and tokenizer
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(model_name)

from unsloth.chat_templates import get_chat_template

### Had to Add for Phi-4-Unsloth##
#tokenizer = get_chat_template(
#    tokenizer,
#    chat_template = "phi-4",
#)
#####END###

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# Function to prepare test set from CSV
def prepare_test_set_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Extract ground truth stress labels for metrics calculation
    ground_truth = df.groupby('Para_index')['Stress'].first().tolist()

    # Group sentences by paragraph (Para_index) and create the input text
    grouped = df.groupby('Para_index').apply(lambda x: {
        "paragraph": " ".join(x['Column_of_sentences'].tolist()), 
        "sentence_numbers": list(x['Seq_inside_para'])
    }).tolist()

    # Prepare messages for the inference
    messages = []
    for group in grouped:
        instruction = "Stress is defined as a reaction to extant and future demands and pressures,3 which can be positive in moderation. Categorize the input post as ‘Stressful’ or ‘Not Stressful’. Do not return anything other than one of these two labels."
        paragraph = group['paragraph']
        messages.append({
            "role": "user",
            "content": f"Instruction: {instruction}\nInput: {paragraph}\n"
        })

    return messages, ground_truth

# Function for paragraph-wise inference using Llama 3.2
def run_inference(messages):
    predictions = []
    # Loop through each message (each paragraph)
#    messages = messages[:50]
    for message in messages:
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        FastLanguageModel.for_inference(model)  # Enable faster inference

        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

        # Text streamer for real-time output display
        text_streamer = TextStreamer(tokenizer)

        # Generate output for this paragraph
        outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256)

        # Process output
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('\n\n')[-1].replace('\'', '').replace('.', '').strip()		#works for Llama-3.1-8b-Instruct and Llama-3.2-3B-Instruct
#        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('\n ')[-1].replace('\'', '').replace('.', '').strip()		#Works for mistral-7b-Instruct
#        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('assistant')[-1].split('<|im_end|>')[0].strip()	#Qwen-2.5-7B-Instruct, Phi-4-unsloth
        print('\nOT:', output_text)
        print('\n')
        # Extract stress label
        if output_text == 'Stressful':
            predictions.append(1)
        elif output_text == 'Not Stressful':
            predictions.append(0)
        else:
            predictions.append(1)

    return predictions

# Function to calculate metrics
def calculate_metrics(ground_truth, predictions):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)
    recall = recall_score(ground_truth, predictions, zero_division=0)
    f1 = f1_score(ground_truth, predictions, zero_division=0)
    conf_matrix = confusion_matrix(ground_truth, predictions)
    class_report = classification_report(ground_truth, predictions, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

# Example usage
file_path = "SACReD_Test.csv"  # Specify your file path
messages, ground_truth = prepare_test_set_from_csv(file_path)
predictions = run_inference(messages)

# Calculate and display metrics
calculate_metrics(ground_truth, predictions)
