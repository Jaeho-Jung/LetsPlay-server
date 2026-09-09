# LetsPlay — 영유아 역할놀이 챗봇 백엔드 서버

> 영유아 언어 발달을 돕는 **실시간 음성 상호작용 역할놀이 챗봇 LetsPlay**의 백엔드 서버입니다.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud_Run_GPU-4285F4?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)

클라이언트(앱)로부터 오디오 스트림을 수신하여 **STT → LLM → TTS** 파이프라인을 거쳐 실시간 피드백 음성을 반환하는 엔드투엔드 음성 AI 서버입니다.

**핵심 성과**:
- Faster-Whisper (CTranslate2 FP16) 기반 STT 서버 구현 — 성능·메모리 비교의 측정 조건과 한계는 아래 벤치마크에 명시
- WebSocket 기반 비동기 파이프라인으로 LLM 응답을 문장 단위로 TTS에 전달하고 음성 청크 스트리밍
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

> STT 완료 후 LLM·TTS를 문장 단위로 연결해 전체 응답 생성이 끝나기 전에 음성 전송을 시작합니다. 단일 STT 실험의 지연 시간은 서버의 종단 간 지연이나 첫 바이트 수신(TTFB)·첫 음성 재생 시간을 의미하지 않으며, 각 지표는 별도 측정이 필요합니다.
>
> 실제 구현: `gpt_service.async_generate_chat_response()`가 문장 단위(`". ! ?"` 기준)로 yield하면, 즉시 `async_generate_tts_response()`로 전달하여 문장이 완성되는 즉시 PCM 청크를 스트리밍합니다.

---

## 2. 핵심 백엔드 최적화

### 아키텍처 피벗: Edge → Cloud

초기 기획은 모바일 앱 내부에서 TFLite 오프라인 추론이었습니다. 그러나 Whisper의 동적 텐서 할당 구조와 연산자 미지원으로 인해 변환된 모델에서 비정상 토큰(`[50258, 50264...]`)이 반복 출력되는 문제가 발생했습니다.

**결정**: 인식 품질과 실시간성을 동시에 확보하기 위해 Docker + **GCP Cloud Run GPU(T4)** 기반 클라우드 서버사이드 추론으로 전환했습니다.

### 추론 엔진 마이그레이션

기록된 지연 시간과 서버 런타임 적용성을 바탕으로 **Faster-Whisper (CTranslate2 FP16)** 을 도입했습니다.

[연구 README의 GPU 엔진 비교](https://github.com/Jaeho-Jung/LetsPlay-ai-research/blob/966a94e3218608e98a01364cb5a1a7c5e2235342/README.md#실험-1-단일-추론-최적화--cpu-vs-gpu-비교)에 남은 기존 실험 기록입니다. Google Colab T4 GPU에서 동일한 오디오 파일과 `elmenwol/whisper-small_aihub_child`, 배치 1로 각 엔진 로드 후 **워밍업 없이 첫 함수 호출**을 측정했습니다. 오디오 길이는 기록되지 않았고, 전처리·디코딩 구현과 생성 옵션의 동등성도 검증되지 않아 확정적인 속도 개선 배수나 엔진 순위로 해석하지 않습니다.

| 엔진 | 첫 호출 시간 | 메모리 측정 |
|---|:---:|---|
| Direct Inference (FP16) | 6.02s | PyTorch allocator 기준 모델 477.8 MB, 피크 549.3 MB |
| Faster-Whisper (CTranslate2) | 0.80s | **재측정 필요** |

> 기존 CTranslate2 메모리 값은 `torch.cuda.memory_allocated()`로 수집되어 PyTorch 밖의 CUDA 할당을 반영하지 못했을 가능성이 큽니다. 따라서 VRAM 절감률은 제시하지 않으며, NVML 또는 `nvidia-smi`로 프로세스 GPU 메모리를 재측정해야 합니다.
>
> CPU/GPU 기록은 워밍업·반복 횟수·측정 경로가 달라 직접적인 속도 배수 계산에 사용하지 않습니다. 실제 성능·VRAM 우위는 동일한 오디오·생성 옵션·워밍업·반복 측정 조건에서 재검증해야 하며, 비용 효율과 동시 처리 용량도 기존 기록만으로 입증되지 않습니다.

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

별도 CTranslate2 FP16 실험에서 동일한 오디오 경로를 반복 사용해 **요청 50건**을 처리한 기록입니다. Baseline은 순차 실행하고 나머지는 각 구현의 큐·동시 실행 방식을 사용했습니다. 모델 로딩과 네트워크 I/O는 제외했으며, P95는 각 요청 함수의 시작부터 완료까지의 경과 시간입니다. 오디오 길이와 독립적인 워밍업 횟수는 기록되지 않아 단일 추론 표나 실제 서버의 동시 요청 성능과 직접 비교하지 않습니다.

| 전략 | QPS | P95 지연 | 기록 내 관찰 |
|---|:---:|:---:|:---:|
| **Baseline (순차)** | 5.28 | **0.259s** | 가장 낮은 P95 |
| num_workers (CT2) | 6.62 | 7.394s | 높은 처리량, 긴 꼬리 지연 |
| Async Queue | 2.18 | 22.016s | 낮은 처리량, 긴 꼬리 지연 |

**전략 해석**: 기록상 Baseline의 P95가 가장 낮지만, 이를 서비스 지연 보장이나 외부 큐가 항상 불필요하다는 근거로 사용하지 않습니다. Rate Limiting은 운영 전략으로 검토하되, 위 표의 Baseline 측정값을 Rate Limiting 적용 효과로 해석하지 않습니다. 실제 SLA 판단 전에는 오디오 길이·워밍업·생성 옵션·동시 요청 시작 조건·서버 I/O를 고정해 재측정해야 합니다.

**확장 후속 검토안 (부하 검증 필요)**: Async Queue는 backpressure와 자원 사용량 제어를 위한 선택지입니다. 고부하 처리량, OOM 방지 효과와 확장성은 후속 검증이 필요합니다.
```
Redis Async Queue → GPU당 Worker Process → 수평 확장 검토
큐 포화 시 요청 거절·재시도 정책 검토
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
