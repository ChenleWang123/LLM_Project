import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "model/mistral_lora_adapter"

print("1. Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# Test input
test_text = "<s>[INST] What is the most popular children's animation in the US? [/INST]"
inputs = tokenizer(test_text, return_tensors="pt").to("cuda")

# --- Step A: Test WITHOUT LoRA ---
print("\n2. Testing WITHOUT LoRA...")
with torch.no_grad():
    base_output = base_model.generate(**inputs, max_new_tokens=10)
    base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)
    print(f"Base Model Output: {base_text.split('[/INST]')[-1].strip()}")

# --- Step B: Test WITH LoRA ---
print("\n3. Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("4. Testing WITH LoRA...")
with torch.no_grad():
    lora_output = model.generate(**inputs, max_new_tokens=10)
    lora_text = tokenizer.decode(lora_output[0], skip_special_tokens=True)
    print(f"LoRA Model Output: {lora_text.split('[/INST]')[-1].strip()}")

# --- Final Check ---
if base_text == lora_text:
    print("\n❌ 结果完全一致：LoRA 确实没有起到任何作用。")
else:
    print("\n✅ 结果不同：LoRA 已经生效，它成功改变了模型的预测逻辑！")