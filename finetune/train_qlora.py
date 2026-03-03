import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# Configuration variables
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "../data/finetune_data/mixed_train.jsonl"
OUTPUT_DIR = "../model/mistral_lora_adapter"


def train():
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading formatted dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # === SPEED UP HACK 1: Subsample the dataset ===
    # Shuffle and pick only 1000 examples (about half of your data)
    num_samples = min(1000, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    print(f"Subsampled dataset to {num_samples} examples for faster training.")

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)

    print("Setting up LoRA configuration...")
    # === SPEED UP HACK 2: Simplify LoRA modules ===
    # Reduce rank (r=8) and only target q_proj and v_proj
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # === SPEED UP HACK 3: Reduce epochs and batch accumulation ===
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,  # Reduced from 4
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=5,  # Log more frequently to see progress
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=1,  # Reduced from 3 to 1
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=True,
    )

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        # === SPEED UP HACK 4: Cut max_seq_length by half ===
        max_seq_length=256,  # Drastically reduces computation without Flash Attention
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting fast fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Fast fine-tuning completed!")


if __name__ == "__main__":
    train()