from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch, re
import pandas as pd

# Load the model and tokenizer
model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(model_name)

# Function to prepare test set from CSV
def prepare_test_set_from_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Group sentences by paragraph (Para_index) and create the input text and labels
    grouped = df.groupby('Para_index').apply(lambda x: {
        "sentences": list(x['Column_of_sentences'].tolist()),  # List of sentences
        "sentence_numbers": list(x['Seq_inside_para']),        # Sentence numbers
    }).tolist()

    # Prepare messages for the inference
    messages = []
    for group in grouped:
        # Few-shot examples
        examples = (
            "### Example 1\n"
            "Input:\n"
            "1. I have been struggling with work deadlines lately.\n"
            "2. My family has been very supportive.\n"
            "3. My boss constantly criticizes my work.\n"
            "Output: [1, 3]\n\n"
            "### Example 2\n"
            "Input:\n"
            "1. The weather is beautiful today.\n"
            "2. I went for a walk in the park and felt peaceful.\n"
            "3. Nothing stressful has been happening lately.\n"
            "Output: []\n\n"
        )

        # Add the actual task
        instruction = (
            "Stress is defined as a reaction to extant and future demands and pressures, which can be positive in moderation. "
            "You are assigned the task of identifying the sentence numbers in a stressful post that indicate the causes for the associated stress. "
            "If the post is non-stressful, return NOTHING. Please return the sentence numbers only. "
            "Fullstop or exclamation marks can be considered as sentence delimiters. "
            "Do not return any sentence numbers outside the valid range of sentences provided below. Here are a couple of examples for your understanding:\n"
        )
        
        # Format sentences as a numbered list
        sentences = "\n".join([f"{num}. {sent.strip()}" for num, sent in zip(group['sentence_numbers'], group['sentences'])])

        # Add the task input
        messages.append({
            "role": "user",
            "content": f"Instruction: {instruction}\n{examples}\n\nInput:\n{sentences}\n"
        })
    return messages

# Function for paragraph-wise inference using Llama 3.2
def run_inference(messages):
    results = []; preds = []
#    messages = messages[:10]
    # Loop through each message (each paragraph)
    for message in messages:
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        FastLanguageModel.for_inference(model)  # Enable faster inference

        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

        # Text streamer for real-time output display
        text_streamer = TextStreamer(tokenizer)

        # Generate output for this paragraph
        outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)

        # Process output
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split('<|eot_id|>')[0]

        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, output_text)
        print('\n\nOT: ', matches)
        if len(matches) == 2:
            print([])
            preds.append([])
        elif len(matches) > 2:
            print([matches[2]])
            preds.append([matches[-1]])
        print('\n')
    print(preds)
    return results, preds

# Example usage
file_path = "SACReD_Test.csv"  # Specify your file path
messages = prepare_test_set_from_csv(file_path)
results, predictions = run_inference(messages)

def format_predictions(predictions):
    """
    Formats predictions to ensure all numbers are properly extracted as integers.
    
    Args:
    predictions (list): List of lists where each sublist contains predicted numbers as strings.
    
    Returns:
    list: List of formatted predictions where each sublist contains integers.
    """
    formatted_predictions = []
    for sublist in predictions:
        if isinstance(sublist, list):
            # Handle comma-separated numbers in sublist
            formatted_sublist = []
            for item in sublist:
                if isinstance(item, str):
                    formatted_sublist.extend(
                        [int(num.strip()) for num in item.split(',') if num.strip().isdigit()]
                    )
                elif isinstance(item, int):
                    formatted_sublist.append(item)
            formatted_predictions.append(formatted_sublist)
        else:
            formatted_predictions.append([])
    return formatted_predictions


# Reformat predictions
cleaned_preds = format_predictions(predictions)
print("\n\nCleaned Predictions: ", cleaned_preds[:10])

import pandas as pd

def extract_gold_labels(file_path):
    """
    Extract gold labels for stress-causing sentences from a CSV file.
    Ensures all paragraphs are included, even if they have no stress-causing sentences.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        list of lists: Gold labels formatted as a list of sentence numbers for each paragraph.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Group all paragraphs by Para_index
    all_paras = df['Para_index'].unique()
    
    # Extract stress-causing sentence numbers
    gold_labels = (
        df[df['Stress_cause'] == 1]
        .groupby('Para_index')['Seq_inside_para']
        .apply(list)
        .reindex(all_paras, fill_value=[])
        .tolist()
    )

    return gold_labels

# Example usage
file_path = "SACReD_Test.csv"  # Replace with the path to your CSV file
gold_labels = extract_gold_labels(file_path)

print("\n\nGold Labels: ", gold_labels[:10])

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
#
#def calculate_per_para_metrics(gold_labels, predictions):
#    """
#    Calculates per-paragraph precision, recall, F1-score, and accuracy, and computes their averages.
#
#    Parameters:
#    - gold_labels: List of lists where each sublist contains sentence numbers (integers) marked as stressful in the gold labels.
#    - predictions: List of lists where each sublist contains sentence numbers (integers) marked as stressful by the model.
#
#    Returns:
#    - average_metrics: Dictionary containing the average precision, recall, F1-score, and accuracy.
#    """
#    per_para_metrics = []
#    
#    for gold, pred in zip(gold_labels, predictions):
#        # Determine the maximum sentence number in the paragraph
#        max_sentence = max(gold) if gold else (max(pred) if pred else 0)
#        
#        # Create binary arrays for gold and predicted labels
#        gold_binary = [1 if i in gold else 0 for i in range(1, max_sentence + 1)]
#        pred_binary = [1 if i in pred else 0 for i in range(1, max_sentence + 1)]
#        
#        # Calculate precision, recall, F1-score, and accuracy for this paragraph
#        precision = precision_score(gold_binary, pred_binary, zero_division=0)
#        recall = recall_score(gold_binary, pred_binary, zero_division=0)
#        f1 = f1_score(gold_binary, pred_binary, zero_division=0)
#        accuracy = accuracy_score(gold_binary, pred_binary)
#        
#        per_para_metrics.append({
#            "precision": precision,
#            "recall": recall,
#            "f1": f1,
#            "accuracy": accuracy
#        })
#    
#    # Calculate average metrics across all paragraphs
#    average_metrics = {
#        "average_precision": np.mean([m["precision"] for m in per_para_metrics]),
#        "average_recall": np.mean([m["recall"] for m in per_para_metrics]),
#        "average_f1": np.mean([m["f1"] for m in per_para_metrics]),
#        "average_accuracy": np.mean([m["accuracy"] for m in per_para_metrics]),
#    }
#    
#    return per_para_metrics, average_metrics
#
### Example Usage
##gold_labels = [[2, 3], [], [1, 4], [1, 2, 3]]  # Gold labels for paragraphs
##predictions = [[2], [], [4], [1, 3]]           # Model predictions
#
#per_para_metrics, average_metrics = calculate_per_para_metrics(gold_labels, cleaned_preds)
#
## Print per-paragraph metrics
#for i, metrics in enumerate(per_para_metrics, 1):
#    print(f"Paragraph {i} Metrics: {metrics}")
#
## Print average metrics
#print("\nAverage Metrics:")
#for metric, value in average_metrics.items():
#    print(f"{metric}: {value:.4f}")

def calculate_total_sentences_per_para(df):
    """
    Calculate the total number of sentences in each paragraph based on Seq_inside_para.

    Args:
    df (pd.DataFrame): DataFrame containing 'Para_ID' and 'Seq_inside_para' columns.

    Returns:
    list: Total sentences per paragraph.
    """
    # Group by Para_ID and find the max Seq_inside_para for each paragraph
    total_sentences_per_para = (
        df.groupby("Para_index")["Seq_inside_para"].max().sort_index().tolist()
    )
    return total_sentences_per_para

def calculate_metrics(predictions, gold_labels, total_sentences_per_para):
    """
    Calculate per-para and average metrics (precision, recall, F1, accuracy).
    
    Args:
    predictions (list): List of lists of predicted sentence numbers.
    gold_labels (list): List of lists of actual sentence numbers.
    total_sentences_per_para (list): List of total sentences in each paragraph.

    Returns:
    dict: Per-para metrics and average metrics.
    """
#    assert len(predictions) == len(gold_labels) == len(total_sentences_per_para), \
#        "Mismatch in lengths of predictions, gold labels, and total sentences."

    all_metrics = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for pred, gold, total_sentences in zip(predictions, gold_labels, total_sentences_per_para):
        pred_set = set(pred)
        gold_set = set(gold)
        all_sentences = set(range(1, total_sentences + 1))
        
        tp = len(pred_set & gold_set)  # True Positives
        fp = len(pred_set - gold_set)  # False Positives
        fn = len(gold_set - pred_set)  # False Negatives
        tn = len(all_sentences - (pred_set | gold_set))  # True Negatives
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        accuracy = (tp + tn) / total_sentences if total_sentences > 0 else 0.0

        all_metrics.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        })

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    # Compute averages
    avg_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    avg_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if avg_precision + avg_recall > 0 else 0.0
    avg_accuracy = (total_tp + total_tn) / sum(total_sentences_per_para) if sum(total_sentences_per_para) > 0 else 0.0

    return {
        "per_paragraph_metrics": all_metrics,
        "average_metrics": {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "average_accuracy": avg_accuracy,
        },
    }


df = pd.read_csv(file_path)
total_sentences_per_para = calculate_total_sentences_per_para(df)

metrics = calculate_metrics(cleaned_preds, gold_labels, total_sentences_per_para)
print(metrics)
