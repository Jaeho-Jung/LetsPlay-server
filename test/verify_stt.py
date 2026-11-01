"""Transcribe repository WAV samples locally; no API calls or model downloads."""
import json
import time
import wave
from pathlib import Path

from src.whisper_service import WhisperService


def main():
    started = time.perf_counter()
    service = WhisperService(device="cpu")
    result = {"device": "cpu", "compute_type": service.compute_type,
              "model_load_seconds": time.perf_counter() - started, "samples": []}
    for path in sorted(Path(__file__).parent.glob("sample_input_*.wav")):
        with wave.open(str(path)) as audio:
            duration = audio.getnframes() / audio.getframerate()
        started = time.perf_counter()
        try:
            text = service.transcribe_sync(path.read_bytes())
            row = {"sample": path.name, "audio_seconds": duration, "transcript": text,
                   "success": bool(text.strip())}
        except Exception as exc:
            row = {"sample": path.name, "audio_seconds": duration, "success": False,
                   "error_type": type(exc).__name__}
        row["elapsed_seconds"] = time.perf_counter() - started
        result["samples"].append(row)
        Path("../letsplay_stt_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if not all(row["success"] for row in result["samples"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
