import base64
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


INDEX_TTS_REPO = Path("/mnt/moark-models/github/index-tts")
MODEL_DIR = Path("/mnt/moark-models/IndexTTS-2")
CONFIG_PATH = INDEX_TTS_REPO / "checkpoints" / "config.yaml"
DEFAULT_REF_AUDIO = INDEX_TTS_REPO / "emo_sad.wav"
OUTPUT_PATH = Path("/data/exam/tts_output.wav")

DEFAULT_TEXT = "模力方舟语音合成任务已经完成，祝大家实验顺利。"
DEFAULT_REF_TEXT = "参考音频用于提供说话人音色和情感。"

app = FastAPI(title="Task 5 TTS API")
tts_model = None


def prepare_runtime() -> None:
    """Use writable cache directories instead of the read-only model tree."""
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/indextts-cache")
    os.environ.setdefault("HF_HOME", "/tmp/indextts-hf-home")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/indextts-matplotlib")

    for path in (
        Path(os.environ["XDG_CACHE_HOME"]),
        Path(os.environ["HF_HOME"]),
        Path(os.environ["MPLCONFIGDIR"]),
        OUTPUT_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    if str(INDEX_TTS_REPO) not in sys.path:
        sys.path.insert(0, str(INDEX_TTS_REPO))

    src = INDEX_TTS_REPO / "indextts" / "utils" / "tagger_cache"
    dst = Path("/tmp/indextts-cache/tagger_cache")
    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)
    os.environ.setdefault("INDextts_TAGGER_CACHE", str(dst))
    os.environ.setdefault("INDEXTTS_TAGGER_CACHE", str(dst))


def validate_paths() -> None:
    for path in (INDEX_TTS_REPO, MODEL_DIR, CONFIG_PATH, DEFAULT_REF_AUDIO):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")


def load_tts_model():
    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2(
        cfg_path=str(CONFIG_PATH),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )


def inspect_wav(path: Path) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        framerate = wav.getframerate()
        return {
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "framerate": framerate,
            "frames": frames,
            "duration": frames / float(framerate),
        }


async def parse_request(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        data: Dict[str, Any] = {
            "input": form.get("input") or DEFAULT_TEXT,
            "model": form.get("model") or "IndexTTS-2",
            "ref_text": form.get("ref_text") or DEFAULT_REF_TEXT,
        }

        upload = form.get("ref_audio")
        if upload is not None and hasattr(upload, "read"):
            suffix = Path(getattr(upload, "filename", "") or "ref.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await upload.read())
                data["ref_audio"] = tmp.name
        else:
            data["ref_audio"] = str(DEFAULT_REF_AUDIO)

        return data

    if "application/json" in content_type:
        body = await request.json()
        return {
            "input": body.get("input") or DEFAULT_TEXT,
            "model": body.get("model") or "IndexTTS-2",
            "ref_text": body.get("ref_text") or DEFAULT_REF_TEXT,
            "ref_audio": body.get("ref_audio") or str(DEFAULT_REF_AUDIO),
        }

    return {
        "input": DEFAULT_TEXT,
        "model": "IndexTTS-2",
        "ref_text": DEFAULT_REF_TEXT,
        "ref_audio": str(DEFAULT_REF_AUDIO),
    }


def synthesize(text: str, ref_audio: str, ref_text: str) -> None:
    if tts_model is None:
        raise RuntimeError("TTS model is not loaded")

    tts_model.infer(
        spk_audio_prompt=ref_audio,
        text=text,
        output_path=str(OUTPUT_PATH),
        emo_audio_prompt=ref_audio,
        prompt_text=ref_text,
        verbose=True,
    )


def response_payload() -> Dict[str, Any]:
    audio_bytes = OUTPUT_PATH.read_bytes()
    return {
        "data": [
            {
                "b64_json": base64.b64encode(audio_bytes).decode("utf-8"),
            }
        ],
        "b64_json": base64.b64encode(audio_bytes).decode("utf-8"),
        "path": str(OUTPUT_PATH),
        "audio_format": "wav",
        "wav_info": inspect_wav(OUTPUT_PATH),
    }


@app.on_event("startup")
def startup() -> None:
    global tts_model

    prepare_runtime()
    validate_paths()
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("Loading IndexTTS2...")
    tts_model = load_tts_model()
    print("IndexTTS2 loaded.")


@app.get("/")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": "IndexTTS-2",
        "output_path": str(OUTPUT_PATH),
    }


@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    temp_ref_audio: Optional[str] = None

    try:
        data = await parse_request(request)
        ref_audio = str(data["ref_audio"])
        if ref_audio != str(DEFAULT_REF_AUDIO):
            temp_ref_audio = ref_audio

        synthesize(
            text=str(data["input"]),
            ref_audio=ref_audio,
            ref_text=str(data["ref_text"]),
        )

        return JSONResponse(response_payload())

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
            },
        )

    finally:
        if temp_ref_audio:
            try:
                os.remove(temp_ref_audio)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "task5_tts_server:app",
        host="0.0.0.0",
        port=8188,
        reload=False,
    )
