import os
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, UploadFile
from qwen_asr import Qwen3ASRModel


MODEL_PATH = "/mnt/moark-models/Qwen3-ASR-1.7B"
OUTPUT_PATH = "/data/exam/asr_output.txt"

app = FastAPI(title="Task 4 ASR API")

model = None


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return "Chinese"

    lang = language.strip()

    mapping = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "cn": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
    }

    return mapping.get(lang.lower(), lang)


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


@app.on_event("startup")
def load_model():
    global model

    os.makedirs("/data/exam", exist_ok=True)

    print("Loading Qwen3-ASR model...")
    model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    print("Qwen3-ASR model loaded.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(default="Qwen3-ASR-1.7B", alias="model"),
    language: Optional[str] = Form(default="zh"),
):
    global model

    os.makedirs("/data/exam", exist_ok=True)

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        lang = normalize_language(language)

        print(f"Received audio: {file.filename}")
        print(f"model: {model_name}")
        print(f"language: {language} -> {lang}")
        print(f"temp path: {tmp_path}")

        try:
            result = model.transcribe(
                tmp_path,
                language=lang,
            )
        except TypeError:
            result = model.transcribe(tmp_path)

        text = extract_text(result)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(text)

        print("识别结果:")
        print(text)
        print(f"Saved to {OUTPUT_PATH}")

        return {"text": text}

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "task4_asr_server:app",
        host="0.0.0.0",
        port=8188,
        reload=False,
    )
