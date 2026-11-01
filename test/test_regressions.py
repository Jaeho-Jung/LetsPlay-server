import asyncio
import importlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.whisper_service import WhisperService
from src.gpt_service import GPTService
from src.utils.utils import MODEL_NAME


def test_model_path_matches_repository():
    assert Path(MODEL_NAME).is_absolute()
    assert (Path(MODEL_NAME) / "config.json").is_file()


def test_wav_bytes_are_passed_as_file_object(monkeypatch):
    model = MagicMock()
    model.transcribe.return_value = (
        iter([SimpleNamespace(text="안녕하세요")]),
        SimpleNamespace(language="ko", language_probability=1.0, duration=1.0),
    )
    WhisperService.reset()
    monkeypatch.setattr(WhisperService, "_load_model", lambda self: model)
    service = WhisperService(device="cpu")
    assert asyncio.run(service.transcribe_audio(b"RIFF-sample")) == "안녕하세요"
    received = model.transcribe.call_args.args[0]
    assert isinstance(received, io.BytesIO)
    assert received.read() == b"RIFF-sample"
    WhisperService.reset()


def test_each_gpt_service_has_clients(monkeypatch):
    import src.gpt_service as module
    monkeypatch.setattr(module, "OpenAI", MagicMock())
    monkeypatch.setattr(module, "AsyncOpenAI", MagicMock())
    first, second = GPTService("test-key"), GPTService("test-key")
    assert hasattr(first, "async_client") and hasattr(second, "async_client")


def test_streaming_honors_token_limit():
    calls = []

    async def chunks():
        for text in ["안녕", "하세요.", " 반가워요!"]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    async def create(**kwargs):
        calls.append(kwargs)
        return chunks()

    service = GPTService.__new__(GPTService)
    service.async_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    async def collect():
        return [x async for x in service.async_generate_chat_response([], max_tokens=42)]

    assert asyncio.run(collect()) == ["안녕하세요.", "반가워요!"]
    assert calls[0]["max_tokens"] == 42


@pytest.fixture
def app_module(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-for-network")
    monkeypatch.setattr(WhisperService, "_load_model", lambda self: MagicMock())
    WhisperService.reset()
    module = importlib.import_module("main")

    class Speech:
        async def transcribe_audio(self, data):
            return data.decode()

    class Chat:
        async def async_generate_chat_response(self, context):
            yield "응답:" + "|".join(x["content"] for x in context if x["role"] == "user")

        async def async_generate_tts_response(self, sentence):
            yield b"pcm-chunk"

    monkeypatch.setattr(module, "whisper_service", Speech())
    monkeypatch.setattr(module, "gpt_service", Chat())
    module.conversation_contexts.clear()
    yield module
    module.conversation_contexts.clear()
    WhisperService.reset()


def exchange(ws, text):
    ws.send_bytes(text.encode())
    stt, llm, audio, end = ws.receive_json(), ws.receive_json(), ws.receive_bytes(), ws.receive_json()
    assert stt == {"type": "STT", "content": text}
    assert audio == b"pcm-chunk"
    assert end == {"type": "END", "content": ""}
    return llm["content"]


def test_websocket_protocol_and_session_isolation(app_module):
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/roleplay/stream") as one:
            assert exchange(one, "우주") == "응답:우주"
            with client.websocket_connect("/roleplay/stream") as two:
                assert exchange(two, "바다") == "응답:바다"
                assert exchange(one, "별") == "응답:우주|별"
        assert not app_module.conversation_contexts


def test_reset_clears_active_session_history(app_module):
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/roleplay/stream") as ws:
            exchange(ws, "이전 대화")
            assert client.post("/roleplay/reset_conversation").status_code == 200
            assert exchange(ws, "새 대화") == "응답:새 대화"


def test_error_closes_connection_and_releases_history(app_module):
    from starlette.websockets import WebSocketDisconnect

    class FailingSpeech:
        async def transcribe_audio(self, data):
            raise ValueError("invalid WAV")

    app_module.whisper_service = FailingSpeech()
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/roleplay/stream") as ws:
            ws.send_bytes(b"invalid")
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
        assert not app_module.conversation_contexts


def test_demo_plays_short_audio_response(monkeypatch):
    import sys
    written = []
    stream = SimpleNamespace(write=written.append, stop_stream=lambda: None, close=lambda: None)
    player = SimpleNamespace(open=lambda **kwargs: stream, terminate=lambda: None)
    monkeypatch.setitem(sys.modules, "pyaudio", SimpleNamespace(PyAudio=lambda: player, paInt16=8))
    spec = importlib.util.spec_from_file_location("demo_client", Path(__file__).with_name("test.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def play():
        queue = asyncio.Queue()
        for chunk in (b"one", b"two", None):
            await queue.put(chunk)
        await module.play_audio(queue)

    asyncio.run(play())
    assert written == [b"one", b"two"]
