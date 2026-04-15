# 🛰️ 위성 SAR 이미지 변화 탐지 + 탐색 최적화 플랫폼

위성 SAR 이미지에서 변화 영역을 자동 탐지하고, 강화학습으로 관측 우선순위를 최적화하며, LLM이 분석 리포트를 자동 생성하는 풀스택 시스템

## 시스템 구조

```
사용자 (Gradio UI)
    │
    ▼
Nest.js 백엔드 (TypeScript)  ─── PostgreSQL (메타데이터)
    │                              Redis (Bull 메시지 큐)
    │ HTTP 호출
    ▼
Python ML 서비스 (FastAPI)
    ├── PyTorch U-Net ─── 변화 탐지
    ├── PPO (Stable-Baselines3) ─── 탐색 최적화
    └── Gemma + LangChain RAG ─── 리포트 생성
```

## 기술 스택

| 영역 | 기술 |
|------|------|
| **백엔드** | Nest.js, TypeScript, TypeORM, PostgreSQL, Redis, Bull |
| **ML/DL** | PyTorch, U-Net (CNN), Stable-Baselines3 (PPO) |
| **LLM** | Ollama (Gemma2), LangChain, FAISS, HuggingFace Embeddings |
| **인프라** | Docker Compose, FastAPI, Gradio |

## 주요 기능

1. **변화 탐지**: 두 시점 SAR 이미지를 U-Net CNN으로 비교하여 변화 영역 자동 탐지
2. **탐색 최적화**: PPO 강화학습 에이전트가 제한된 관측 횟수 내에서 변화율 높은 영역을 우선 발견
3. **자동 리포트**: Gemma LLM + RAG 파이프라인으로 탐지 결과 기반 자연어 분석 리포트 생성
4. **데이터 파이프라인**: Nest.js → Bull 큐 → Python ML 호출 → PostgreSQL 저장의 비동기 처리

## 실행 방법

```bash
# 1. DB/Redis 실행
docker-compose up -d

# 2. Python ML 서비스
cd python-ml-service && python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python data/download_data.py
python models/train_unet.py
python rl/train_rl.py
uvicorn main:app --port 8001

# 3. Nest.js 백엔드
cd nest-backend && npm install && npm run start:dev

# 4. Gradio UI
cd gradio-ui && pip install -r requirements.txt && python app.py
```

## 프로젝트 구조

```
satellite-project/
├── docker-compose.yml          # PostgreSQL + MongoDB + Redis
├── python-ml-service/          # Python ML 마이크로서비스
│   ├── main.py                 # FastAPI 서버 (포트 8001)
│   ├── models/
│   │   ├── unet.py            # U-Net 변화 탐지 모델
│   │   └── train_unet.py      # 모델 학습 스크립트
│   ├── rl/
│   │   ├── env.py             # Gymnasium 커스텀 환경
│   │   └── train_rl.py        # PPO 에이전트 학습
│   └── llm/
│       └── report_generator.py # RAG 리포트 생성
├── nest-backend/               # Nest.js 백엔드 (포트 3000)
│   └── src/
│       ├── main.ts
│       ├── app.module.ts
│       ├── analysis/          # 분석 CRUD + 파이프라인
│       └── queue/             # Bull 큐 프로세서
└── gradio-ui/                 # Gradio 데모 UI (포트 7860)
    └── app.py
```
