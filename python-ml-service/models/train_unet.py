"""
2주차: U-Net 모델 학습
합성 SAR 데이터로 변화 탐지 모델 훈련
GPU 없으면 Google Colab에서 실행 (런타임 > 런타임 유형 변경 > GPU)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from unet import UNet

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "images")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


class SARChangeDataset(Dataset):
    def __init__(self, split="train"):
        self.dir = os.path.join(DATA_DIR, split)
        self.files = sorted(os.listdir(os.path.join(self.dir, "before")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        before = np.array(
            Image.open(os.path.join(self.dir, "before", fname))
        ).astype(np.float32) / 255.0
        after = np.array(
            Image.open(os.path.join(self.dir, "after", fname))
        ).astype(np.float32) / 255.0
        mask = np.array(
            Image.open(os.path.join(self.dir, "mask", fname))
        ).astype(np.float32) / 255.0

        # 2채널 입력: [before, after]
        x = np.stack([before, after], axis=0)
        y = mask[np.newaxis, ...]  # [1, H, W]

        return torch.tensor(x), torch.tensor(y)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"학습 디바이스: {device}")

    # 데이터 로드
    train_ds = SARChangeDataset("train")
    test_ds = SARChangeDataset("test")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4)

    print(f"학습 데이터: {len(train_ds)}쌍, 테스트 데이터: {len(test_ds)}쌍")

    # 모델, 옵티마이저, 손실함수
    model = UNet(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 학습 루프
    num_epochs = 30
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # 검증
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                test_loss += criterion(pred, y_batch).item()

        avg_test = test_loss / len(test_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train:.4f} | Test Loss: {avg_test:.4f}")

        # 최고 모델 저장
        if avg_test < best_loss:
            best_loss = avg_test
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "unet_best.pth"))
            print(f"  → 모델 저장됨 (loss: {best_loss:.4f})")

    print(f"\n학습 완료! 최고 모델: {os.path.join(MODEL_DIR, 'unet_best.pth')}")


if __name__ == "__main__":
    train()
