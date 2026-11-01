"""Run repository WAV samples through the actual STT, OpenAI LLM and TTS services."""
import argparse
import json
import platform
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from main import app

    results = {"environment": {"platform": platform.system(), "architecture": platform.machine(),
                               "device": "cpu", "compute_type": "float32"}, "samples": []}
    with TestClient(app) as client:
        for path in sorted(Path(__file__).parent.glob("sample_input_*.wav")):
            with wave.open(str(path)) as audio:
                duration = audio.getnframes() / audio.getframerate()
            result = {"sample": path.name, "audio_seconds": duration, "events": [], "pcm_bytes": 0}
            started = time.perf_counter()
            try:
                with client.websocket_connect("/roleplay/stream") as ws:
                    ws.send_bytes(path.read_bytes())
                    while True:
                        message = ws.receive()
                        if message["type"] == "websocket.close":
                            raise RuntimeError("server closed before END")
                        if message.get("bytes") is not None:
                            result["pcm_bytes"] += len(message["bytes"])
                            result.setdefault("first_pcm_seconds", time.perf_counter() - started)
                        elif message.get("text") is not None:
                            event = json.loads(message["text"])
                            result["events"].append(event)
                            if event["type"] == "STT":
                                result["stt_seconds"] = time.perf_counter() - started
                            if event["type"] == "END":
                                break
                result["success"] = result["pcm_bytes"] > 0
            except Exception as exc:
                result["success"] = False
                result["error_type"] = type(exc).__name__
            result["total_seconds"] = time.perf_counter() - started
            results["samples"].append(result)
            Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
            print(json.dumps(result, ensure_ascii=False), flush=True)
    if not all(row["success"] for row in results["samples"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
