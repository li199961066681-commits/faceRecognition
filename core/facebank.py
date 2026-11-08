# -*- coding: utf-8 -*-
"""facebank.py —— 人脸库：注册 / 搜索 / 统计 / 删除

落盘结构（相对项目根目录，自动创建）：
  data/facebank.json            所有特征与元信息
  data/registered/<姓名>/*.jpg  注册时保存的 112x112 对齐人脸（供界面预览）

策略：best-of-k —— 每人保留多张照片的特征向量，
识别时取该人所有特征中的最高相似度（不平均，保留角度多样性）。
"""
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import FaceDetector
from .extractor import FaceExtractor

ROOT = Path(__file__).resolve().parent.parent      # 项目根目录
DATA_DIR = ROOT / "data"
BANK_JSON = DATA_DIR / "facebank.json"
PHOTO_DIR = DATA_DIR / "registered"


class FaceBank:
    def __init__(self):
        self.detector = FaceDetector()
        self.extractor = FaceExtractor()
        # {姓名: {"features": [np(512), ...], "photos": [路径,...], "time": str}}
        self.people = {}
        self._load()

    # ---------------- 持久化 ----------------
    def _load(self):
        if not BANK_JSON.exists():
            return
        raw = json.loads(BANK_JSON.read_text(encoding="utf-8"))
        for name, info in raw.items():
            self.people[name] = {
                "features": [np.asarray(f, dtype=np.float32) for f in info["features"]],
                "photos": info.get("photos", []),
                "time": info.get("time", ""),
            }

    def _save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw = {name: {"features": [f.tolist() for f in info["features"]],
                      "photos": info["photos"], "time": info["time"]}
               for name, info in self.people.items()}
        BANK_JSON.write_text(json.dumps(raw, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    # ---------------- 注册 ----------------
    def register(self, name, bgr, is_frontal=True):
        """注册一张照片。成功返回 {"ok": True, ...}，失败返回 {"ok": False, "msg": 原因}。
        is_frontal 为预留参数：当前 best-of-k 策略不区分正侧脸，保留接口便于扩展。"""
        name = (name or "").strip()
        if not name:
            return {"ok": False, "msg": "姓名不能为空"}

        bbox, aligned = self.detector.get_main_face(bgr)
        if bbox is None:
            return {"ok": False, "msg": "未检测到人脸（按已验证成功的构图重拍）"}

        feat = self.extractor.extract(aligned)                # (512,) 已归一化

        # 防重复：与该人已有特征几乎相同则跳过
        person = self.people.setdefault(name, {"features": [], "photos": [], "time": ""})
        if person["features"]:
            dup = float((np.stack(person["features"]) @ feat).max())
            if dup > 0.995:
                return {"ok": False, "msg": f"疑似重复照片（相似度 {dup:.3f}），已跳过"}

        person_dir = PHOTO_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{int(time.time() * 1000)}.jpg"              # 毫秒时间戳避免重名
        ok, buf = cv2.imencode(".jpg", aligned)
        if ok:
            buf.tofile(str(person_dir / fname))               # 中文路径安全写盘
            person["photos"].append(str(person_dir / fname))

        person["features"].append(feat)
        person["time"] = time.strftime("%Y-%m-%d %H:%M")
        self._save()
        return {"ok": True, "msg": "注册成功", "n": len(person["features"]),
                "bbox": bbox, "photo": str(person_dir / fname) if ok else None}

    # ---------------- 搜索 ----------------
    def search(self, feat):
        """best-of-k：返回 [(姓名, 该人最高余弦相似度), ...]，按分数降序。"""
        ranked = []
        for name, info in self.people.items():
            if not info["features"]:
                continue
            scores = np.stack(info["features"]) @ feat        # 归一化后内积=余弦
            ranked.append((name, float(scores.max())))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ---------------- 管理 ----------------
    def remove(self, name):
        if name not in self.people:
            return False
        del self.people[name]
        self._save()
        shutil.rmtree(PHOTO_DIR / name, ignore_errors=True)
        return True

    def all_names(self):
        return list(self.people.keys())

    def stats(self):
        return {n: len(i["features"]) for n, i in self.people.items()}
