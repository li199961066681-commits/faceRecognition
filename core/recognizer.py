# -*- coding: utf-8 -*-
"""recognizer.py —— 识别决策：检测 → 提特征 → 查库 → 按阈值判定

判定规则：库中所有人的最高分里取第一名，
  score >= threshold → known（本人）
  score <  threshold → stranger（陌生人）
"""

DEFAULT_THRESHOLD = 0.50


def recognize(bgr, bank, threshold=DEFAULT_THRESHOLD):
    """返回 dict：status ∈ {no_face, known, stranger}，含 bbox / score / top3。"""
    bbox, aligned = bank.detector.get_main_face(bgr)
    if bbox is None:
        return {"status": "no_face", "msg": "未检测到人脸"}

    if not bank.all_names():
        return {"status": "no_face", "msg": "人脸库为空，请先在注册页添加人员"}

    feat = bank.extractor.extract(aligned)
    ranked = bank.search(feat)
    name, score = ranked[0]
    top3 = [(n, round(s, 4)) for n, s in ranked[:3]]

    if score >= threshold:
        return {"status": "known", "name": name, "score": round(float(score), 4),
                "bbox": bbox, "top3": top3}
    return {"status": "stranger", "name": None, "score": round(float(score), 4),
            "bbox": bbox, "top3": top3}
