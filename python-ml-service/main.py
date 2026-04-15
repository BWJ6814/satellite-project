"""
Python ML 서비스 API 서버
Nest.js 백엔드에서 HTTP로 호출하는 마이크로서비스
포트: 8001
"""
import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

app = FastAPI(title="위성 이미지 분석 ML 서비스", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


# ── 응답 모델 ──
class ChangeDetectionResult(BaseModel):
    changes: list[dict]
    total_change_rate: float
    num_regions: int


class PriorityResult(BaseModel):
    priorities: list[dict]


class ReportResult(BaseModel):
    report: str


# ── 1. 변화 탐지 API ──
@app.post("/api/detect-changes", response_model=ChangeDetectionResult)
async def detect_changes(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
):
    """2단계: 두 SAR 이미지를 비교하여 변화 영역 탐지"""
    before_img = np.array(Image.open(io.BytesIO(await before.read())).convert("L"))
    after_img = np.array(Image.open(io.BytesIO(await after.read())).convert("L"))

    # U-Net 모델 로드 시도, 없으면 간단한 차분 방식 사용
    model_path = os.path.join(MODEL_DIR, "unet_best.pth")
    if os.path.exists(model_path):
        from models.unet import UNet
        model = UNet(in_channels=2, out_channels=1)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        h, w = before_img.shape
        x = np.stack([before_img / 255.0, after_img / 255.0], axis=0).astype(np.float32)
        x_tensor = torch.tensor(x).unsqueeze(0)

        with torch.no_grad():
            pred = model(x_tensor).squeeze().numpy()

        change_mask = (pred > 0.5).astype(np.uint8)
    else:
        # 폴백: 단순 차분 방식
        diff = np.abs(before_img.astype(float) - after_img.astype(float))
        threshold = np.mean(diff) + 2 * np.std(diff)
        change_mask = (diff > threshold).astype(np.uint8)

    # 변화 영역 분석
    from scipy import ndimage
    try:
        labeled, num_features = ndimage.label(change_mask)
    except ImportError:
        labeled = change_mask
        num_features = 1

    changes = []
    total_pixels = change_mask.shape[0] * change_mask.shape[1]
    changed_pixels = int(np.sum(change_mask))

    for i in range(1, min(num_features + 1, 20)):
        region = (labeled == i)
        if np.sum(region) < 10:
            continue
        ys, xs = np.where(region)
        changes.append({
            "id": i,
            "x": int(np.mean(xs)),
            "y": int(np.mean(ys)),
            "area_pixels": int(np.sum(region)),
            "change_rate": round(float(np.sum(region)) / total_pixels * 100, 2),
            "bbox": {
                "x1": int(xs.min()), "y1": int(ys.min()),
                "x2": int(xs.max()), "y2": int(ys.max())
            }
        })

    return ChangeDetectionResult(
        changes=changes,
        total_change_rate=round(changed_pixels / total_pixels * 100, 2),
        num_regions=len(changes),
    )


# ── 2. 탐색 우선순위 API ──
@app.post("/api/prioritize")
async def prioritize_scan(changes: list[dict]):
    """3단계: PPO 에이전트로 탐색 우선순위 계산"""
    model_path = os.path.join(MODEL_DIR, "ppo_satellite.zip")

    if os.path.exists(model_path):
        from stable_baselines3 import PPO
        from rl.env import SatelliteScanEnv

        # 변화 맵 생성
        change_map = np.zeros((8, 8), dtype=np.float32)
        for c in changes:
            gx = min(int(c["x"] / 32), 7)
            gy = min(int(c["y"] / 32), 7)
            change_map[gy, gx] = c.get("change_rate", 0.5) / 100.0

        env = SatelliteScanEnv(change_map=change_map)
        model = PPO.load(model_path)

        obs, _ = env.reset()
        visit_order = []
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            pos = env.agent_pos.copy()
            visit_order.append({"step": step, "y": pos[0], "x": pos[1], "reward": round(reward, 3)})
            done = terminated or truncated
            step += 1

        return {"priorities": visit_order, "total_steps": step, "found": info["changes_found"]}
    else:
        # 폴백: 변화율 기준 정렬
        sorted_changes = sorted(changes, key=lambda c: c.get("change_rate", 0), reverse=True)
        priorities = [{"rank": i + 1, **c} for i, c in enumerate(sorted_changes)]
        return {"priorities": priorities, "method": "fallback_sort"}


# ── 3. LLM 리포트 API ──
@app.post("/api/generate-report", response_model=ReportResult)
async def generate_report(request: dict):
    """5단계: LLM으로 분석 리포트 생성"""
    from llm.report_generator import get_generator

    gen = get_generator()

    analysis_results = request.get("analysis_results", [])
    question = request.get("question", "전체 분석 결과를 요약해주세요")

    if analysis_results:
        gen.add_analysis_results(analysis_results)

    try:
        report = gen.generate_report(question)
    except Exception as e:
        report = (
            f"LLM 리포트 생성 실패: {str(e)}\n"
            f"Ollama가 실행 중인지 확인하세요 (ollama serve)\n\n"
            f"분석 데이터 요약:\n"
            + "\n".join([f"- {r.get('region', 'N/A')}: 변화율 {r.get('change_rate', 0)}%"
                         for r in analysis_results])
        )

    return ReportResult(report=report)


# ── 헬스체크 ──
@app.get("/health")
async def health():
    return {"status": "ok", "models": {
        "unet": os.path.exists(os.path.join(MODEL_DIR, "unet_best.pth")),
        "ppo": os.path.exists(os.path.join(MODEL_DIR, "ppo_satellite.zip")),
    }}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
