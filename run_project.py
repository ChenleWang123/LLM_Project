import pandas as pd
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import re  # Add this to the top of your script if not already there


# Import our custom RAG retriever
# Ensure rag/retriever.py is accessible from this script
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))
from rag.retriever import CulturalRetriever

# ================= Configuration Area =================
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "model/mistral_lora_adapter"  # Path to your fine-tuned weights

FILES = {
    "test_mcq": "data/test_dataset_mcq.csv",
    "test_saq": "data/test_dataset_saq.csv"
}


# ====================================================

def setup_model():
    """Load the base model and apply the LoRA adapter."""
    print(f"Loading base model: {MODEL_ID}...")

    # 4-bit quantization config to save VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Applying LoRA adapter from: {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    return tokenizer, model


def run_mcq(model, tokenizer, retriever):
    """Run Multiple Choice Question task with RAG."""
    print("\n--- Running MCQ Task (RAG + Logits Strategy) ---")
    df_test = pd.read_csv(FILES["test_mcq"])

    # Pre-compute token IDs for A, B, C, D
    choices = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]

    results = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        # 1. Retrieve background knowledge
        context = retriever.search(row['prompt'], top_k=3)

        # 2. Construct the RAG Prompt
        instruction = "You are a cross-cultural expert. Use the provided context to help you answer. Read the question carefully and output ONLY the letter of the correct option (A, B, C, or D)."
        prompt = f"<s>[INST] {instruction}\n\nContext:\n{context}\n\nQuestion:\n{row['prompt']} [/INST]"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # 3. Predict using logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            scores = [logits[i].item() for i in choice_ids]
            best_idx = scores.index(max(scores))
            best_choice = choices[best_idx]

        entry = {
            "MCQID": row['MCQID'],
            "A": False, "B": False, "C": False, "D": False
        }
        entry[best_choice] = True
        results.append(entry)

    # Save results
    out_df = pd.DataFrame(results)
    out_df = out_df[["MCQID", "A", "B", "C", "D"]]
    out_df.to_csv("mcq_prediction.tsv", sep='\t', index=False)
    print("✅ MCQ predictions saved to mcq_prediction.tsv")




def run_saq(model, tokenizer, retriever):
    """Run Short Answer Question task with RAG and aggressive post-processing."""
    print("\n--- Running SAQ Task (RAG + Zero-Shot Strategy) ---")
    df_test = pd.read_csv(FILES["test_saq"])

    results = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        question = row['en_question']

        # 1. Retrieve background knowledge
        context = retriever.search(question, top_k=3)

        # 2. Construct the RAG Prompt
        instruction = "You are a cross-cultural expert. Use the provided context to help you answer. Answer the question with a single entity, phrase, or short words. Do not use full sentences."
        final_prompt = f"<s>[INST] {instruction}\n\nContext:\n{context}\n\nQuestion:\n{question} [/INST]"

        inputs = tokenizer(final_prompt, return_tensors="pt").to("cuda")

        # 3. Generate short answer
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        # 4. Decode and AGGRESSIVELY clean the output
        full_ans = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract everything after [/INST]
        raw_ans = full_ans.split("[/INST]")[-1].strip()

        # Use Regex to chop off hallucinations, lists, and explanations
        # This splits the string at the first occurrence of [, (, |, /, #, or newline
        clean_ans = re.split(r'\[|\(|\||/|#|\n', raw_ans)[0].strip()

        # Remove any trailing weird punctuation that might be left
        clean_ans = clean_ans.rstrip(":-;,")

        results.append({"ID": row['ID'], "answer": clean_ans})

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv("saq_prediction_cleaned.tsv", sep='\t', index=False)
    print("✅ Cleaned SAQ predictions saved to saq_prediction_cleaned.tsv")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ Error: GPU not found. This script requires a GPU.")
    else:
        # Initialize the RAG Retriever
        print("Initializing RAG Retriever...")
        retriever = CulturalRetriever()

        # Initialize the Fine-tuned Model
        tokenizer, model = setup_model()

        # Run both tasks
        run_mcq(model, tokenizer, retriever)
        run_saq(model, tokenizer, retriever)

        print("\n🎉 All tasks completed! You can now submit the .tsv files to Codabench.")