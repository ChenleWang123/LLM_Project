import os
import pandas as pd
import torch
import ast
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# ================= 配置区域 =================
# 模型路径 (首次运行会自动下载)
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
# 文件路径 (确保这4个csv文件和脚本在同一目录)
FILES = {
    "train_mcq": "data/train_dataset_mcq.csv",
    "test_mcq": "data/test_dataset_mcq.csv",
    "train_saq": "data/train_dataset_saq.csv",
    "test_saq": "data/test_dataset_saq.csv"
}
# ===========================================

def setup_model():
    print(f"Loading model: {MODEL_ID}...")

    # 4-bit 量化配置 (为了在 HPC 省显存并提速)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

def run_mcq(model, tokenizer):
    print("\n--- Running MCQ Task (Logits Strategy) ---")
    df_test = pd.read_csv(FILES["test_mcq"])

    # 预计算 A, B, C, D 的 token ID
    choices = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]

    results = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        # 构造 Prompt: 只要问题和选项
        prompt = f"[INST] Read the question. Choose the correct option (A, B, C, or D).\n\n{row['prompt']} [/INST] The answer is"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)
            # 获取最后一个 token 的预测概率分布
            logits = outputs.logits[0, -1, :]

            # 只比较 A, B, C, D 四个 token 的分数
            scores = [logits[i].item() for i in choice_ids]
            best_idx = scores.index(max(scores)) # 找到分数最高的索引
            best_choice = choices[best_idx]

        # 格式化输出 (True/False)
        entry = {
            "MCQID": row['MCQID'],
            "A": False, "B": False, "C": False, "D": False
        }
        entry[best_choice] = True
        results.append(entry)

    # 保存结果
    out_df = pd.DataFrame(results)
    out_df = out_df[["MCQID", "A", "B", "C", "D"]] # 确保列顺序
    out_df.to_csv("mcq_prediction.tsv", sep='\t', index=False)
    print("✅ MCQ predictions saved to mcq_prediction.tsv")

def run_saq(model, tokenizer):
    print("\n--- Running SAQ Task (Few-Shot Strategy) ---")
    df_train = pd.read_csv(FILES["train_saq"])
    df_test = pd.read_csv(FILES["test_saq"])

    # 1. 准备 Few-Shot 样本 (从训练集提取正确答案)
    def get_clean_ans(s):
        try:
            d = ast.literal_eval(s)
            return d[0]['en_answers'][0] if d and 'en_answers' in d[0] else "unknown"
        except: return "unknown"

    df_train['clean_ans'] = df_train['annotations'].apply(get_clean_ans)

    # 随机取 3 个例子作为示范
    examples = df_train.sample(3)
    few_shot_prompt = ""
    for _, row in examples.iterrows():
        few_shot_prompt += f"Question: {row['en_question']}\nAnswer: {row['clean_ans']}\n\n"

    results = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        question = row['en_question']

        # 2. 构造 Prompt: 强指令 + 例子 + 当前问题
        instruction = "Answer the question with a single entity, phrase, or number. Do not use full sentences."
        final_prompt = f"[INST] {instruction}\n\n{few_shot_prompt}Question: {question}\nAnswer: [/INST]"

        inputs = tokenizer(final_prompt, return_tensors="pt").to("cuda")

        # 3. 生成 (限制长度，防止废话)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15, # 关键：限制只能输出短语
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False # 贪婪搜索，最稳定
            )

        ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 [/INST] 之后的内容
        ans = ans.split("[/INST]")[-1].strip().split("\n")[0]

        results.append({"ID": row['ID'], "answer": ans})

    # 保存结果
    out_df = pd.DataFrame(results)
    out_df.to_csv("saq_prediction.tsv", sep='\t', index=False)
    print("✅ SAQ predictions saved to saq_prediction.tsv")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ Error: GPU not found. This script requires a GPU.")
    else:
        tokenizer, model = setup_model()
        run_mcq(model, tokenizer)
        run_saq(model, tokenizer)
        print("\n🎉 All tasks completed! Zip the .tsv files and submit.")