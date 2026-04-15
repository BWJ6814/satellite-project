"""
2주차: 위성 이미지 데이터 다운로드
OSCD (Onera Satellite Change Detection) 데이터셋 대신
간단한 합성 데이터를 생성하여 프로토타입 진행
실제 Sentinel-1 데이터는 https://scihub.copernicus.eu/ 에서 무료 다운로드 가능
"""
import os
import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "images")


def generate_synthetic_sar_pair(size=256, num_changes=5):
    """
    SAR 위성 이미지 쌍을 합성으로 생성
    실제 프로젝트에서는 Sentinel-1 GRD 데이터로 교체
    """
    np.random.seed(42)

    # 배경 생성 (SAR 이미지 특성: 스펙클 노이즈)
    before = np.random.rayleigh(scale=40, size=(size, size)).astype(np.uint8)
    after = before.copy()

    # 변화 영역 생성 (건물 신축, 삼림 변화 등 시뮬레이션)
    change_mask = np.zeros((size, size), dtype=np.uint8)

    for _ in range(num_changes):
        cx, cy = np.random.randint(30, size - 30, 2)
        w, h = np.random.randint(15, 40, 2)
        x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
        x2, y2 = min(size, cx + w // 2), min(size, cy + h // 2)

        # after 이미지에 변화 적용
        intensity_change = np.random.randint(80, 180)
        after[y1:y2, x1:x2] = np.clip(
            after[y1:y2, x1:x2].astype(int) + intensity_change, 0, 255
        ).astype(np.uint8)
        change_mask[y1:y2, x1:x2] = 255

    return before, after, change_mask


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 학습용 데이터 생성 (50쌍)
    train_dir = os.path.join(DATA_DIR, "train")
    os.makedirs(os.path.join(train_dir, "before"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "after"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "mask"), exist_ok=True)

    print("학습 데이터 생성 중...")
    for i in range(50):
        before, after, mask = generate_synthetic_sar_pair(
            size=256, num_changes=np.random.randint(2, 8)
        )
        Image.fromarray(before).save(
            os.path.join(train_dir, "before", f"{i:04d}.png")
        )
        Image.fromarray(after).save(
            os.path.join(train_dir, "after", f"{i:04d}.png")
        )
        Image.fromarray(mask).save(
            os.path.join(train_dir, "mask", f"{i:04d}.png")
        )

    # 테스트용 데이터 생성 (10쌍)
    test_dir = os.path.join(DATA_DIR, "test")
    os.makedirs(os.path.join(test_dir, "before"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "after"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "mask"), exist_ok=True)

    print("테스트 데이터 생성 중...")
    for i in range(10):
        before, after, mask = generate_synthetic_sar_pair(
            size=256, num_changes=np.random.randint(2, 8)
        )
        Image.fromarray(before).save(
            os.path.join(test_dir, "before", f"{i:04d}.png")
        )
        Image.fromarray(after).save(
            os.path.join(test_dir, "after", f"{i:04d}.png")
        )
        Image.fromarray(mask).save(
            os.path.join(test_dir, "mask", f"{i:04d}.png")
        )

    print(f"완료! 학습 50쌍, 테스트 10쌍 생성됨")
    print(f"저장 위치: {DATA_DIR}")
    print("\n※ 실제 SAR 데이터로 교체하려면:")
    print("  1. https://scihub.copernicus.eu/ 에서 Sentinel-1 GRD 다운로드")
    print("  2. 같은 지역, 다른 시점 이미지 2장을 before/after 폴더에 넣기")


if __name__ == "__main__":
    main()
