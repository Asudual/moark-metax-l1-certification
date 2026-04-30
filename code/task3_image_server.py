import base64
import io
import os
from typing import List, Optional

import torch
from diffusers import ZImagePipeline
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image


MODEL_PATH = "/mnt/moark-models/Z-Image-Turbo"
OUTPUT_PATH = "/data/exam/image_output.png"

DEFAULT_PROMPT = (
    "一只可爱的橘色小猫坐在窗台上，窗外是夕阳余晖洒满的城市天际线，"
    "温暖的光线透过玻璃照在猫咪的毛发上，营造出宁静祥和的氛围。"
    "高清摄影风格，细节丰富。"
)

app = FastAPI(title="Task 3 Image Generation API")

pipe = None


class ImageGenerationRequest(BaseModel):
    model: Optional[str] = "Qwen-Image-2512"
    prompt: str = DEFAULT_PROMPT
    size: Optional[str] = "1024x1024"
    num_inference_steps: Optional[int] = 8
    guidance_scale: Optional[float] = 0.0
    seed: Optional[int] = 42
    n: Optional[int] = 1


class ImageData(BaseModel):
    b64_json: str


class ImageGenerationResponse(BaseModel):
    data: List[ImageData]


def parse_size(size: str):
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        return 1024, 1024


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.on_event("startup")
def load_model():
    global pipe

    os.makedirs("/data/exam", exist_ok=True)

    print("Loading Z-Image-Turbo pipeline...")
    pipe = ZImagePipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    print("Image generation pipeline loaded.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
def generate_image(req: ImageGenerationRequest):
    global pipe

    width, height = parse_size(req.size or "1024x1024")
    n = max(1, int(req.n or 1))

    results = []

    for i in range(n):
        seed = int(req.seed or 42) + i
        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.no_grad():
            image = pipe(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=int(req.num_inference_steps or 8),
                guidance_scale=float(req.guidance_scale or 0.0),
                negative_prompt="",
                generator=generator,
            ).images[0]

        if i == 0:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            image.save(OUTPUT_PATH)
            print(f"Saved first image to {OUTPUT_PATH}")

        results.append(ImageData(b64_json=image_to_base64(image)))

    return ImageGenerationResponse(data=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "task3_image_server:app",
        host="0.0.0.0",
        port=8188,
        reload=False,
    )
