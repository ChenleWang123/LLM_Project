import pandas as pd
import ast
import os
import json

MCQ_CSV = "../data/train_dataset_mcq.csv"
SAQ_CSV = "../data/train_dataset_saq.csv"
# Unify the file name
CORPUS_FILE = "../data/corpus/master_knowledge.txt"


def extract_knowledge():
    os.makedirs(os.path.dirname(CORPUS_FILE), exist_ok=True)
    knowledge_chunks = []

    # Extract from MCQ
    df_mcq = pd.read_csv(MCQ_CSV)
    for _, row in df_mcq.iterrows():
        prompt = str(row.get('prompt', ''))
        choices_str = str(row.get('choices', '{}'))
        answer_idx = str(row.get('answer_idx', ''))

        if pd.isna(prompt) or pd.isna(answer_idx): continue

        try:
            choices = json.loads(choices_str)
            correct_answer = choices.get(answer_idx, "")
            fact = f"Cross-cultural fact: For the question '{prompt}', the correct answer is '{correct_answer}'."
            knowledge_chunks.append(fact)
        except Exception:
            continue

    # Extract from SAQ
    df_saq = pd.read_csv(SAQ_CSV)
    for _, row in df_saq.iterrows():
        question = str(row.get('en_question', ''))
        annotations_str = str(row.get('annotations', '[]'))

        try:
            annotations = ast.literal_eval(annotations_str)
            if annotations and isinstance(annotations, list):
                correct_answer = annotations[0]['en_answers'][0]
                fact = f"Cross-cultural fact: Regarding '{question}', the accurate entity or phrase is '{correct_answer}'."
                knowledge_chunks.append(fact)
        except Exception:
            continue

    print(f"\n--- Starting CSV Data Extraction ---")
    # Use "w" mode to initialize the master file and write CSV facts first
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for chunk in knowledge_chunks:
            f.write(chunk.replace("\n", " ") + "\n")

    print(f"✅ {len(knowledge_chunks)} CSV facts written to: {CORPUS_FILE}")


if __name__ == "__main__":
    extract_knowledge()