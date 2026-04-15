"""
2주차: PyTorch U-Net 모델
두 시점 SAR 이미지를 입력받아 변화 영역 마스크를 출력
→ 서울다이나믹스 어필: PyTorch 기반 알고리즘 구현
"""
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    입력: 2채널 (before 이미지 + after 이미지, 각 그레이스케일)
    출력: 1채널 (변화 마스크, 0~1 확률)
    """

    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        # 인코더 (다운샘플링)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # 바틀넥
        self.bottleneck = DoubleConv(512, 1024)

        # 디코더 (업샘플링)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # 최종 출력
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 바틀넥
        b = self.bottleneck(self.pool(e4))

        # 디코더 + 스킵 연결
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


if __name__ == "__main__":
    # 모델 구조 확인
    model = UNet(in_channels=2, out_channels=1)
    x = torch.randn(1, 2, 256, 256)
    y = model(x)
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {y.shape}")
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
