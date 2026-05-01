import os
import shutil
import sys
import wave
from pathlib import Path

import torch


INDEX_TTS_REPO = Path("/mnt/moark-models/github/index-tts")
MODEL_DIR = Path("/mnt/moark-models/IndexTTS-2")
CONFIG_PATH = INDEX_TTS_REPO / "checkpoints" / "config.yaml"
REF_AUDIO = INDEX_TTS_REPO / "emo_sad.wav"
OUTPUT_PATH = Path("/data/exam/tts_output.wav")

DEFAULT_TEXT = "模力方舟语音合成任务已经完成，祝大家实验顺利。"
DEFAULT_REF_TEXT = "参考音频用于提供说话人音色和情感。"


def prepare_runtime() -> None:
    """Make IndexTTS imports and writable cache paths work in the exam image."""
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


def validate_paths() -> None:
    for path in (INDEX_TTS_REPO, MODEL_DIR, CONFIG_PATH, REF_AUDIO):
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")


def copy_tagger_cache_if_needed() -> None:
    """Avoid writing tagger cache under the read-only /mnt/moark-models tree."""
    src = INDEX_TTS_REPO / "indextts" / "utils" / "tagger_cache"
    dst = Path("/tmp/indextts-cache/tagger_cache")

    if src.exists() and not dst.exists():
        shutil.copytree(src, dst)

    os.environ.setdefault("INDextts_TAGGER_CACHE", str(dst))
    os.environ.setdefault("INDEXTTS_TAGGER_CACHE", str(dst))


def load_tts_model():
    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2(
        cfg_path=str(CONFIG_PATH),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
    )


def synthesize(tts, text: str, output_path: Path) -> None:
    """Call IndexTTS2 with the argument set used in the successful run."""
    tts.infer(
        spk_audio_prompt=str(REF_AUDIO),
        text=text,
        output_path=str(output_path),
        emo_audio_prompt=str(REF_AUDIO),
        prompt_text=DEFAULT_REF_TEXT,
        verbose=True,
    )


def inspect_wav(path: Path) -> dict:
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


def main() -> None:
    prepare_runtime()
    validate_paths()
    copy_tagger_cache_if_needed()

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("Loading IndexTTS2...")
    tts = load_tts_model()

    print("Running TTS inference...")
    synthesize(tts, DEFAULT_TEXT, OUTPUT_PATH)

    info = inspect_wav(OUTPUT_PATH)
    print("saved to:", OUTPUT_PATH)
    print("wav info:", info)


if __name__ == "__main__":
    main()
