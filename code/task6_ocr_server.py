import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoProcessor


MODEL_PATH = "/mnt/moark-models/FireRed-OCR"
OUTPUT_PATH = "/data/exam/ocr_output.txt"
PROMPT = "请识别图片中的文字，只输出识别结果，不要添加解释。"

app = FastAPI(title="Task 6 OCR API")

processor = None
model = None


def get_model_class():
    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq

            return AutoModelForVision2Seq


def build_inputs(image_path: str, image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            return processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except Exception:
            return processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
    except Exception:
        return processor(
            text=[PROMPT],
            images=[image],
            padding=True,
            return_tensors="pt",
        )


def decode_generated_text(generated_ids, input_length: int) -> str:
    generated = generated_ids[:, input_length:]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def run_ocr(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = build_inputs(image_path, image).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    text = decode_generated_text(generated_ids, inputs.input_ids.shape[-1])

    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    return text


@app.on_event("startup")
def load_model() -> None:
    global processor, model

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Required model path does not exist: {MODEL_PATH}")

    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("Loading FireRed-OCR processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading FireRed-OCR model...")
    model_class = get_model_class()
    model = model_class.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    print("FireRed-OCR loaded.")


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "FireRed-OCR",
        "output_path": OUTPUT_PATH,
    }


@app.post("/v1/vision/ocr")
async def ocr(file: UploadFile = File(...)):
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        tmp.write(await file.read())

    try:
        text = run_ocr(tmp_path)
        return {"text": text}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "task6_ocr_server:app",
        host="0.0.0.0",
        port=8188,
        reload=False,
    )
