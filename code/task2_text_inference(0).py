from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re

MODEL_PATH = "/mnt/moark-models/Qwen3-8B"
OUTPUT_PATH = "/data/exam/text_inference.txt"

def clean_output(text: str) -> str:
    """
    清理 Qwen3 可能输出的 <think> 思考内容。
    如果存在 </think>，保留 </think> 后面的正文。
    如果只有 <think> 没有闭合，则尽量移除 <think> 标签本身。
    """
    text = text.strip()

    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()

    text = text.replace("<think>", "").replace("</think>", "").strip()

    return text

def main():
    os.makedirs("/data/exam", exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    prompt = (
        "请直接写一段100个字符以上的中文科幻短篇小说。"
        "要求：只输出小说正文，不要解释写作思路，不要列提纲，不要出现<think>。"
        "故事需要有人物、场景、冲突和结尾。"
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    result = clean_output(result)

    print("Result:")
    print(result)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result.strip())

    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()