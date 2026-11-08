"""人脸特征提取：对齐后的 112x112 人脸 -> 512 维特征向量（L2 归一化）。

模型：ArcFace + ResNet18(IR-SE) 预训练权重 arcface_resnet18.pth（约 90MB）。
首次运行自动下载；若直连失败，把 MIRROR 改成镜像前缀（如 "https://ghproxy.net/"）。
"""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from .arcface_r18 import load_arcface_r18

WEIGHT_URL = ("https://github.com/TencentARC/GFPGAN/releases/"
              "download/v0.1.0/arcface_resnet18.pth")
MIRROR = ""                                    # 下载失败时改为镜像前缀，如 "https://ghproxy.net/"


def ensure_weights(weights_path: Path) -> Path:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists() and weights_path.stat().st_size > 10 ** 7:
        return weights_path
    url = MIRROR + WEIGHT_URL
    print(f"开始下载识别模型权重：{url}")
    tmp = weights_path.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, weights_path)
    print("下载完成。")
    return weights_path


class FaceExtractor:
    def __init__(self, weights_path="weights/arcface_resnet18.pth", device="cpu"):
        weights_path = Path(os.environ.get("FACE_WEIGHTS", weights_path))
        self.model = load_arcface_r18(ensure_weights(weights_path), device)
        self.device = device

    @torch.no_grad()
    def extract(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """输入：detector 输出的 112x112 对齐人脸(BGR)；输出：512 维 L2 归一化特征"""
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)          # 该模型用灰度输入
        gray = cv2.resize(gray, (128, 128))                           # 模型输入尺寸 128x128
        x = gray.astype(np.float32) / 127.5 - 1.0                     # 归一化到 [-1, 1]
        x = torch.from_numpy(x)[None, None].to(self.device)           # (1,1,128,128)
        feat = self.model(x).flatten().cpu().numpy()
        return feat / np.linalg.norm(feat)                            # L2 归一化
