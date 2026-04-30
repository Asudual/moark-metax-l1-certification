import os
import torch

from qwen_asr import Qwen3ASRModel


MODEL_PATH = "/mnt/moark-models/Qwen3-ASR-1.7B"
AUDIO_PATH = "/mnt/moark-models/L1_exam/asr_demo.wav"
OUTPUT_PATH = "/data/exam/asr_output.txt"


def extract_text(result):
    if isinstance(result, str):
        return result.strip()

    if isinstance(result, dict):
        for key in ("text", "transcription", "result"):
            if key in result:
                return str(result[key]).strip()

    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ("text", "transcription", "result"):
                if key in first:
                    return str(first[key]).strip()

    return str(result).strip()


def main():
    os.makedirs("/data/exam", exist_ok=True)

    print("Loading Qwen3-ASR model...")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    print("Running ASR...")
    result = model.transcribe(
        AUDIO_PATH,
        language="Chinese",
    )

    print("Raw result:")
    print(result)

    text = extract_text(result)

    print("识别结果:")
    print(text)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
