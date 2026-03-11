# LetsPlay — 영유아 역할놀이 챗봇 백엔드 서버

> 영유아 언어 발달을 돕는 **실시간 음성 상호작용 역할놀이 챗봇 LetsPlay**의 백엔드 서버입니다.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run_GPU-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)

클라이언트(앱)로부터 오디오 스트림을 수신하여 **STT → LLM → TTS** 파이프라인을 거쳐 실시간 피드백 음성을 반환하는 엔드투엔드 음성 AI 서버입니다.

**핵심 성과**:
- CPU 대비 GPU 추론 속도 **42.8배** 향상, Faster-Whisper 도입으로 VRAM **98% 절감** (549MB → 9.9MB)
- WebSocket 기반 비동기 스트리밍 파이프라인으로 **첫 응답 0.5초 내** 음성 재생
- Docker + GCP Cloud Run GPU(T4) 컨테이너 배포

> 모델 파인튜닝, 데이터 증강, 추론 최적화 벤치마크 과정은 [letsplay-ai-research](https://github.com/Jaeho-Jung/LetsPlay-ai-research)에서 확인하실 수 있습니다.

---

## 1. 시스템 아키텍처 및 파이프라인 설계

### 데이터 흐름

```
Client (App)
    │
    │  WebSocket (Audio bytes)
    ▼
┌─────────────────────────────────────────────┐
│               FastAPI Server                │
│                                             │
│  ① STT  Faster-Whisper (CTranslate2 FP16)   │ ← Blocking  (정확도 우선)
│          └─ run_in_executor (thread pool)   │
│                                             │
│  ② LLM  GPT-4o-mini  (stream=True)  ───┐    │ ← Streaming (지연 최소화)
│                                        │    │
│  ③ TTS  OpenAI tts-1 (PCM streaming) ←─┘    │ ← Streaming (문장 단위)
└─────────────────────────────────────────────┘
    │
    │  WebSocket (JSON + Audio PCM chunks)
    ▼
Client (App)
```

### 설계 의사결정: STT는 왜 Blocking인가?

영유아 발화는 단어 사이의 간격(침묵)이 길고 조음이 불명확합니다. Streaming STT로 처리하면 침묵 구간에서 문맥이 단절되어 인식 오류가 급증합니다.

| 처리 방식 | 지연 | 정확도 | 적용 이유 |
|---|:---:|:---:|---|
| **STT: Blocking** | 높음 | 높음 | 발화 완료 후 전체 맥락 보존하여 전사 |
| **LLM+TTS: Streaming** | 낮음 | — | STT 지연 보상 — 문장 단위로 즉시 TTS 전달 |

> STT 지연(~0.8s)은 LLM·TTS 스트리밍으로 상쇄하여 **체감 대기시간(TTFB)을 0.5초 내**로 유지합니다.
>
> 실제 구현: `gpt_service.async_generate_chat_response()`가 문장 단위(`". ! ?"` 기준)로 yield하면, 즉시 `async_generate_tts_response()`로 전달하여 문장이 완성되는 즉시 PCM 청크를 스트리밍합니다.

---

## 2. 핵심 백엔드 최적화

### 아키텍처 피벗: Edge → Cloud

초기 기획은 모바일 앱 내부에서 TFLite 오프라인 추론이었습니다. 그러나 Whisper의 동적 텐서 할당 구조와 연산자 미지원으로 인해 변환된 모델에서 비정상 토큰(`[50258, 50264...]`)이 반복 출력되는 문제가 발생했습니다.

**결정**: 인식 품질과 실시간성을 동시에 확보하기 위해 Docker + **GCP Cloud Run GPU(T4)** 기반 클라우드 서버사이드 추론으로 전환했습니다.

### 추론 엔진 마이그레이션

HuggingFace `pipeline` 오버헤드를 제거하고 **Faster-Whisper (CTranslate2 FP16)** 을 도입했습니다.

| 항목 | HuggingFace Pipeline | Faster-Whisper (CT2) | 개선 |
|---|:---:|:---:|:---:|
| 단건 추론 속도 | 6.02s | **0.80s** | **7.5x** |
| 모델 VRAM | 477.8 MB | **9.9 MB** | **98% 절감** |
| 피크 VRAM | 549.3 MB | **9.9 MB** | **98% 절감** |

> VRAM 98% 절감으로 클라우드 GPU 인스턴스의 비용 효율이 대폭 향상되었으며, 동시 접속 처리의 기반을 마련했습니다.

**서비스 구현**: `WhisperService`는 싱글톤 패턴으로 모델을 한 번만 로드하고, `asyncio.run_in_executor`로 블로킹 추론을 비동기 이벤트 루프에서 논블로킹으로 처리합니다.

```python
# src/whisper_service.py
async def transcribe_audio(self, audio_path: str) -> str:
    loop = asyncio.get_event_loop()
    transcript = await loop.run_in_executor(
        None,               # 기본 ThreadPoolExecutor
        self.transcribe_sync,
        audio_path,
    )
    return transcript
```

---

## 3. 동시성 제어 및 스케일링 전략

N=50 동시 요청 벤치마크 결과를 기반으로 도출한 트래픽 제어 전략입니다.

| 전략 | QPS | P95 지연 | 선택 여부 |
|---|:---:|:---:|:---:|
| **Baseline (순차) + Rate Limiting** | 5.28 | **0.259s** | **현재 적용** |
| num_workers (CT2) | 6.62 | 7.394s | — |
| Async Queue | 2.18 | 22.016s | — |

**현재 (단일 GPU)**: Faster-Whisper의 CTranslate2 내부 커널 스케줄링이 이미 최적화되어 있어 외부 큐는 오버헤드만 추가합니다. **Baseline 순차 처리 + Rate Limiting**으로 P95 0.259s의 가장 낮은 꼬리 지연을 보장합니다.

**스케일업 로드맵 (N > 100)**:
```
Redis Async Queue → GPU당 Worker Process → 처리량 선형 확장
QueueFullError → HTTP 429 (graceful degradation)
```

---

## 4. API 명세

### `POST /roleplay/reset_conversation`

대화 컨텍스트를 초기화합니다.

```json
Response: { "message": "Conversation context reset successfully." }
```

### `WebSocket /roleplay/stream`

실시간 음성 스트리밍 엔드포인트입니다.

```
Client → Server: Audio bytes
Server → Client: { "type": "STT", "content": "전사 텍스트" }
Server → Client: { "type": "LLM", "content": "응답 문장" }
Server → Client: Audio PCM chunks (tts-1, nova voice, 1024 bytes/chunk)
Server → Client: { "type": "END", "content": "" }
```

---

## 5. 배포 및 인프라

### 로컬 실행

```bash
git clone https://github.com/Jaeho-Jung/LetsPlay-server.git
cd LetsPlay-server
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key" > .env
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t letsplay-server .
docker run -d -p 8080:8080 --env-file .env letsplay-server
```

### GCP Cloud Run 배포

```bash
gcloud auth configure-docker
docker tag letsplay-server gcr.io/<project-id>/letsplay-server
docker push gcr.io/<project-id>/letsplay-server

gcloud run deploy letsplay-server \
    --image gcr.io/<project-id>/letsplay-server \
    --platform managed \
    --region asia-northeast3 \
    --allow-unauthenticated \
    --set-env-vars OPENAI_API_KEY=your_key
```

---

## 6. 프로젝트 구조

```
LetsPlay-server/
├── main.py                  # FastAPI 앱 + WebSocket 엔드포인트
├── src/
│   ├── whisper_service.py   # Faster-Whisper 싱글톤 서비스 (STT)
│   ├── gpt_service.py       # GPT-4o-mini 스트리밍 + TTS 서비스
│   ├── log.py               # 로깅 유틸리티
│   ├── utils/
│   │   └── utils.py         # 상수 (모델명, 시스템 프롬프트 등)
│   └── models/              # Pydantic 모델
├── test/                    # 단위 테스트
├── Dockerfile
└── requirements.txt
```

---

## 7. Tech Stack

| Category | Technologies |
|---|---|
| **Backend** | Python 3.10, FastAPI, WebSockets, asyncio |
| **STT** | Faster-Whisper (CTranslate2 FP16), `elmenwol/whisper-small_aihub_child` |
| **LLM** | OpenAI GPT-4o-mini (Streaming) |
| **TTS** | OpenAI tts-1 (PCM Streaming, nova voice) |
| **Infra** | Docker, GCP Cloud Run GPU (T4) |

---

## Author

**정재호 (Jaeho Jung)** — Team Leader

## License

MIT License
