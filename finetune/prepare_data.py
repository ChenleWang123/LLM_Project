import pandas as pd
import json
import ast
import os
import random

# Define file paths based on your current project structure
MCQ_TRAIN_CSV = "../data/train_dataset_mcq.csv"
SAQ_TRAIN_CSV = "../data/train_dataset_saq.csv"
OUTPUT_JSONL = "../data/finetune_data/mixed_train.jsonl"


def process_mcq(df):
    """
    Process Multiple Choice Questions.
    Extract the prompt (which already contains choices) and the exact answer index.
    """
    records = []
    for _, row in df.iterrows():
        prompt = row.get('prompt', '')
        answer = row.get('answer_idx', '')

        if pd.isna(prompt) or pd.isna(answer):
            continue

        # Instruction tailored for MCQ
        instruction = "You are a cross-cultural expert. Read the question carefully and output ONLY the letter of the correct option (A, B, C, or D)."

        # Construct Mistral instruction format
        full_prompt = f"<s>[INST] {instruction}\n\n{prompt} [/INST] {answer} </s>"
        records.append({"text": full_prompt})

    return records


def process_saq(df):
    """
    Process Short Answer Questions.
    Safely evaluate the 'annotations' string to extract the top English answer.
    """
    records = []
    for _, row in df.iterrows():
        question = row.get('en_question', '')
        annotations_str = row.get('annotations', '[]')

        if pd.isna(question):
            continue

        # Safely parse the stringified list of dictionaries
        try:
            annotations = ast.literal_eval(annotations_str)
            # Extract the first English answer from the most frequent annotation
            answer = annotations[0]['en_answers'][0]
        except (ValueError, SyntaxError, IndexError, KeyError):
            continue  # Skip if parsing fails or no answer exists

        # Instruction tailored for SAQ to force short answers
        instruction = "You are a cross-cultural expert. Answer the question with a single entity, phrase, or short words. Do not use full sentences."

        # Construct Mistral instruction format
        full_prompt = f"<s>[INST] {instruction}\n\nQuestion: {question} [/INST] {answer} </s>"
        records.append({"text": full_prompt})

    return records


def main():
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    print("Loading datasets...")
    df_mcq = pd.read_csv(MCQ_TRAIN_CSV)
    df_saq = pd.read_csv(SAQ_TRAIN_CSV)

    print("Processing MCQ data...")
    mcq_records = process_mcq(df_mcq)

    print("Processing SAQ data...")
    saq_records = process_saq(df_saq)

    # Combine and shuffle the dataset so the model doesn't overfit to one task type
    all_records = mcq_records + saq_records
    random.shuffle(all_records)

    # Save to JSONL
    print(f"Saving {len(all_records)} mixed records to {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print("✅ Data preparation completed successfully!")


if __name__ == "__main__":
    main()