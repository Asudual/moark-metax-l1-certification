import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor


MODEL_PATH = "/mnt/moark-models/FireRed-OCR"
IMAGE_PATH = "/mnt/moark-models/L1_exam/ocr_test_image.jpg"
OUTPUT_PATH = "/data/exam/ocr_output.txt"

PROMPT = "请识别图片中的文字，只输出识别结果，不要添加解释。"


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


def validate_paths() -> None:
    for path in (MODEL_PATH, IMAGE_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")


def build_inputs(processor, image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
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


def decode_generated_text(processor, generated_ids, input_length: int) -> str:
    generated = generated_ids[:, input_length:]
    text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def main() -> None:
    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)
    validate_paths()

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading model...")
    model_class = get_model_class()
    model = model_class.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    image = Image.open(IMAGE_PATH).convert("RGB")
    inputs = build_inputs(processor, image).to(model.device)

    print("Running OCR...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    text = decode_generated_text(processor, generated_ids, inputs.input_ids.shape[-1])

    print("OCR result:")
    print(text)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
