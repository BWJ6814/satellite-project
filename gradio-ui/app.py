"""
8주차: Gradio 데모 UI
사용자가 이미지 업로드 → 변화 탐지 → 우선순위 → 리포트까지 한번에 실행
"""
import gradio as gr
import requests
import numpy as np
from PIL import Image
import io
import json

NEST_API = "http://localhost:3000"
ML_API = "http://localhost:8001"


def analyze_images(before_img, after_img, region, question):
    """전체 분석 파이프라인 실행"""

    results = []

    # ── 1. 변화 탐지 ──
    results.append("🔍 1단계: 변화 탐지 중...")
    try:
        before_bytes = io.BytesIO()
        after_bytes = io.BytesIO()
        Image.fromarray(before_img).save(before_bytes, format="PNG")
        Image.fromarray(after_img).save(after_bytes, format="PNG")
        before_bytes.seek(0)
        after_bytes.seek(0)

        detect_resp = requests.post(
            f"{ML_API}/api/detect-changes",
            files={
                "before": ("before.png", before_bytes, "image/png"),
                "after": ("after.png", after_bytes, "image/png"),
            },
            timeout=60,
        )
        detection = detect_resp.json()
        results.append(
            f"  ✅ 변화 영역 {detection['num_regions']}개 발견 "
            f"(전체 변화율: {detection['total_change_rate']}%)"
        )
    except Exception as e:
        detection = {"changes": [], "total_change_rate": 0, "num_regions": 0}
        results.append(f"  ❌ 변화 탐지 실패: {e}")

    # ── 2. 탐색 우선순위 ──
    results.append("\n📊 2단계: 탐색 우선순위 계산 중...")
    try:
        if detection["changes"]:
            prio_resp = requests.post(
                f"{ML_API}/api/prioritize",
                json=detection["changes"],
                timeout=30,
            )
            priorities = prio_resp.json()
            results.append(f"  ✅ 우선순위 계산 완료 ({len(priorities.get('priorities', []))} 스텝)")
        else:
            priorities = {"priorities": []}
            results.append("  ⏭️ 변화 영역 없음, 건너뜀")
    except Exception as e:
        priorities = {"priorities": []}
        results.append(f"  ❌ 우선순위 계산 실패: {e}")

    # ── 3. LLM 리포트 ──
    results.append("\n📝 3단계: LLM 리포트 생성 중...")
    try:
        analysis_for_llm = []
        for i, c in enumerate(detection.get("changes", [])):
            analysis_for_llm.append({
                "region": region or "미지정",
                "x": c.get("x", 0),
                "y": c.get("y", 0),
                "change_rate": c.get("change_rate", 0),
                "area_pixels": c.get("area_pixels", 0),
                "timestamp": "2024-07-15",
                "change_type": "자동 감지",
            })

        report_resp = requests.post(
            f"{ML_API}/api/generate-report",
            json={
                "analysis_results": analysis_for_llm,
                "question": question or "전체 분석 결과를 요약해주세요",
            },
            timeout=120,
        )
        report = report_resp.json().get("report", "리포트 없음")
        results.append("  ✅ 리포트 생성 완료")
    except Exception as e:
        report = f"리포트 생성 실패: {e}"
        results.append(f"  ❌ 리포트 생성 실패: {e}")

    # ── 4. 변화 맵 시각화 ──
    try:
        diff = np.abs(before_img.astype(float) - after_img.astype(float))
        if len(diff.shape) == 3:
            diff = diff.mean(axis=2)
        threshold = diff.mean() + 2 * diff.std()
        mask = (diff > threshold).astype(np.uint8) * 255

        # 원본 위에 빨간색으로 변화 영역 오버레이
        overlay = after_img.copy()
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay] * 3, axis=2)
        overlay[mask > 0] = [255, 50, 50]
        change_map = overlay
    except Exception:
        change_map = after_img

    process_log = "\n".join(results)
    detail_json = json.dumps(detection, indent=2, ensure_ascii=False)

    return change_map, process_log, report, detail_json


# ── Gradio 인터페이스 ──
with gr.Blocks(title="위성 이미지 변화 탐지 플랫폼", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛰️ 위성 SAR 이미지 변화 탐지 플랫폼")
    gr.Markdown(
        "위성 이미지 2장을 업로드하면 → 변화 탐지 → 탐색 우선순위 → "
        "LLM 분석 리포트까지 자동으로 생성됩니다."
    )

    with gr.Row():
        before_input = gr.Image(label="Before 이미지 (이전 시점)", type="numpy")
        after_input = gr.Image(label="After 이미지 (이후 시점)", type="numpy")

    with gr.Row():
        region_input = gr.Textbox(
            label="분석 지역명", placeholder="예: 서울 강남구", value=""
        )
        question_input = gr.Textbox(
            label="LLM에게 질문",
            placeholder="예: 이 지역의 변화 원인을 분석해주세요",
            value="전체 분석 결과를 요약해주세요",
        )

    analyze_btn = gr.Button("🚀 분석 시작", variant="primary", size="lg")

    with gr.Row():
        change_map_output = gr.Image(label="변화 탐지 맵 (빨간색 = 변화 영역)")

    with gr.Row():
        process_log = gr.Textbox(label="처리 과정 로그", lines=10)
        report_output = gr.Textbox(label="LLM 분석 리포트", lines=10)

    detail_output = gr.Code(label="변화 탐지 상세 JSON", language="json")

    analyze_btn.click(
        fn=analyze_images,
        inputs=[before_input, after_input, region_input, question_input],
        outputs=[change_map_output, process_log, report_output, detail_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        "**기술 스택**: PyTorch (U-Net) → PPO (Stable-Baselines3) → "
        "Nest.js + TypeScript (백엔드) → Gemma + LangChain (LLM 리포트) → Gradio (UI)"
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, server_name="0.0.0.0")
