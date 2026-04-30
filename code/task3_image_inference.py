import importlib
import inspect
import json
import os
from pathlib import Path

import torch


OUTPUT_PATH = "/data/exam/image_output.png"

MODEL_CANDIDATES = [
    {
        "name": "Z-Image-Turbo",
        "path": "/mnt/moark-models/Z-Image-Turbo",
        "reason": "Turbo model; prefer it for the first minimal smoke test.",
        "steps": 8,
        "guidance_scale": 0.0,
    },
    {
        "name": "Qwen-Image-2512",
        "path": "/mnt/moark-models/Qwen-Image-2512",
        "reason": "Fallback if ZImagePipeline is not available.",
        "steps": 20,
        "guidance_scale": 4.0,
    },
]

EXPECTED_PIPELINES = {
    "QwenImagePipeline",
    "ZImagePipeline",
}


def load_model_index(model_dir: str) -> dict:
    index_path = Path(model_dir) / "model_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"model_index.json not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_diffusers_module():
    try:
        return importlib.import_module("diffusers")
    except ImportError as exc:
        raise RuntimeError(
            "diffusers is not installed. Install or upgrade it before running this script.\n"
            "Suggested command:\n"
            "  pip install -U 'diffusers>=0.36.0' accelerate transformers sentencepiece safetensors"
        ) from exc


def get_pipeline_class(diffusers_module, class_name: str):
    if class_name not in EXPECTED_PIPELINES:
        raise RuntimeError(
            f"Unsupported or unexpected pipeline class in model_index.json: {class_name!r}. "
            "Do not guess a class name; inspect model_index.json and the installed diffusers package."
        )

    if not hasattr(diffusers_module, class_name):
        installed = getattr(diffusers_module, "__version__", "unknown")
        raise RuntimeError(
            f"Installed diffusers does not export {class_name}.\n"
            f"Installed diffusers version: {installed}\n"
            "The model_index.json files require diffusers 0.36.0.dev0 style pipelines.\n"
            "Suggested command:\n"
            "  pip install -U 'diffusers>=0.36.0' accelerate transformers sentencepiece safetensors"
        )

    return getattr(diffusers_module, class_name)


def build_call_kwargs(pipe, candidate: dict, device: str) -> dict:
    prompt = (
        "A small white robot standing beside a glowing lake at sunrise, "
        "cinematic lighting, detailed, clean composition"
    )

    signature = inspect.signature(pipe.__call__)
    params = signature.parameters

    kwargs = {"prompt": prompt}

    if "height" in params:
        kwargs["height"] = 512
    if "width" in params:
        kwargs["width"] = 512
    if "num_inference_steps" in params:
        kwargs["num_inference_steps"] = candidate["steps"]
    if "guidance_scale" in params:
        kwargs["guidance_scale"] = candidate["guidance_scale"]
    if "true_cfg_scale" in params:
        kwargs["true_cfg_scale"] = candidate["guidance_scale"]
    if "negative_prompt" in params:
        kwargs["negative_prompt"] = ""
    if "generator" in params:
        kwargs["generator"] = torch.Generator(device=device).manual_seed(42)

    return kwargs


def main():
    os.makedirs("/data/exam", exist_ok=True)

    diffusers_module = get_diffusers_module()
    print("diffusers version:", getattr(diffusers_module, "__version__", "unknown"))

    detected = []
    for candidate in MODEL_CANDIDATES:
        model_index = load_model_index(candidate["path"])
        class_name = model_index.get("_class_name")
        detected.append((candidate, class_name))
        print(f"{candidate['name']}: _class_name={class_name}")

    selected = None
    selected_class = None
    selected_error = None

    for candidate, class_name in detected:
        try:
            pipeline_class = get_pipeline_class(diffusers_module, class_name)
            selected = candidate
            selected_class = pipeline_class
            break
        except Exception as exc:
            selected_error = exc
            print(f"Skip {candidate['name']}: {exc}")

    if selected is None or selected_class is None:
        raise RuntimeError(
            "No usable pipeline class was found in the current diffusers installation. "
            "Please install or upgrade diffusers first."
        ) from selected_error

    print("Selected model:", selected["name"])
    print("Selected path:", selected["path"])
    print("Selection reason:", selected["reason"])
    print("Selected pipeline:", selected_class.__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print("device:", device)
    print("dtype:", dtype)

    pipe = selected_class.from_pretrained(
        selected["path"],
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = pipe.to(device)

    call_kwargs = build_call_kwargs(pipe, selected, device)
    print("Call kwargs:", {k: v for k, v in call_kwargs.items() if k != "generator"})

    with torch.no_grad():
        result = pipe(**call_kwargs)

    image = result.images[0]
    image.save(OUTPUT_PATH)
    print("saved to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
