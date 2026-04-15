# 위성 이미지 변화 탐지 플랫폼 - 설치 가이드

## 현재 환경 확인 완료
- Python 3.10.6 ✅
- Node.js v24.14.1 ✅

---

## 0단계: 추가 설치 필요 소프트웨어

### 0-1. Git 설치
- 다운로드: https://git-scm.com/download/win
- 설치 후 확인:
```cmd
git --version✅
```

### 0-2. Docker Desktop 설치✅
- 다운로드: https://www.docker.com/products/docker-desktop/
- 설치 후 재부팅 → Docker Desktop 실행
- 확인:
```cmd
docker --version
docker-compose --version
```
C:\Users\human-26>docker --version
Docker version 29.3.1, build c2be9cc

C:\Users\human-26>docker-compose --version
Docker Compose version v5.1.1

### 0-3. Ollama 설치 (LLM 로컬 실행용)✅
- 다운로드: https://ollama.com/download/windows
- 설치 후 확인:
```cmd
ollama --version
```
- Gemma2 모델 다운로드:
```cmd
ollama pull gemma2
```

### 0-4. Visual Studio Code (코드 에디터)✅
- 다운로드: https://code.visualstudio.com/

---

## 1단계: 프로젝트 폴더 생성 및 이동✅

```cmd
cd C:\Users\human-26
mkdir satellite-project
cd satellite-project
```

이 폴더 안에 이 프로젝트의 모든 파일을 넣습니다.

---

## 2단계: Python 가상환경 및 패키지 설치✅

```cmd
cd C:\Users\human-26\satellite-project\python-ml-service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

설치 확인:
```cmd
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import gymnasium; print('Gymnasium OK')"
python -c "import fastapi; print('FastAPI OK')"
python -c "import langchain; print('LangChain OK')"
```

---

## 3단계: Nest.js 백엔드 설치✅

```cmd
cd C:\Users\human-26\satellite-project\nest-backend
npm install
```

설치 확인:
```cmd
npx ts-node -e "console.log('TypeScript OK')"
```

---

## 4단계: Docker로 DB 실행✅

```cmd
cd C:\Users\human-26\satellite-project
docker-compose up -d postgres mongo redis
```

확인:
```cmd
docker ps
```
→ postgres, mongo, redis 3개 컨테이너가 running 상태여야 합니다.

---

## 5단계: 데이터 다운로드 (2주차)✅

```cmd
cd C:\Users\human-26\satellite-project\python-ml-service
venv\Scripts\activate
python data/download_data.py
```

---

## 6단계: 모델 학습 (2~3주차)

```cmd
# U-Net 학습 (GPU 있으면 로컬, 없으면 Google Colab)
python models/train_unet.py

# PPO 학습 (CPU로 가능)
python rl/train_rl.py
```

---

## 7단계: Python ML 서비스 실행 (4주차~)

```cmd
cd C:\Users\human-26\satellite-project\python-ml-service
venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

확인: 브라우저에서 http://localhost:8001/docs 접속

---

## 8단계: Nest.js 백엔드 실행 (5주차~)

```cmd
cd C:\Users\human-26\satellite-project\nest-backend
npm run start:dev
```

확인: 브라우저에서 http://localhost:3000/api 접속

---

## 9단계: Gradio UI 실행 (8주차)

```cmd
cd C:\Users\human-26\satellite-project\gradio-ui
pip install -r requirements.txt
python app.py
```

확인: 브라우저에서 http://localhost:7860 접속

---

## 전체 한번에 실행 (Docker Compose)

프로젝트 완성 후:
```cmd
cd C:\Users\human-26\satellite-project
docker-compose up --build
```
