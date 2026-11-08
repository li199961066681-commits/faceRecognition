# -*- coding: utf-8 -*-
"""threshold_test.py —— 阈值实验

1. 测试照片按  data/test/<标签>/*.jpg  组织：
   标签 = 已注册人名        → 本人样本
   标签 = 其他名字(陌生人A) → 陌生人样本
2. 运行：python threshold_test.py
3. 输出各阈值下的 准确率 / 误识率(FAR) / 拒识率(FRR)，保存 threshold_curve.png
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")                      # 直接出图，不弹窗口
import matplotlib.pyplot as plt
import numpy as np

from core import FaceBank, imread_cn

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]   # 中文显示
plt.rcParams["axes.unicode_minus"] = False

TEST_DIR = Path(__file__).resolve().parent / "data" / "test"
OUT_PNG = Path(__file__).resolve().parent / "threshold_curve.png"
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.02), 3)


def collect_samples(bank):
    """返回 [(是否本人, 最高分, 预测名, 真实标签, 文件名), ...]"""
    samples = []
    for pdir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
        true_name, is_known = pdir.name, pdir.name in bank.stats()
        for f in sorted(pdir.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = imread_cn(str(f))
            if img is None:
                print(f"[跳过·解码失败] {f.name}")
                continue
            bbox, aligned = bank.detector.get_main_face(img)
            if bbox is None:
                print(f"[跳过·无人脸] {pdir.name}/{f.name}")
                continue
            feat = bank.extractor.extract(aligned)
            ranked = bank.search(feat)
            name, score = ranked[0] if ranked else ("", -1.0)
            samples.append((is_known, score, name, true_name, f.name))
            print(f"{pdir.name}/{f.name}: score={score:.4f} pred={name}"
                  f"（{'本人' if is_known else '陌生人'}）")
    return samples


def metrics(samples, t):
    """返回 (准确率, FAR, FRR)"""
    known = [s for s in samples if s[0]]
    stranger = [s for s in samples if not s[0]]

    correct = sum(1 for _, sc, pred, true, _ in known if sc >= t and pred == true)
    correct += sum(1 for _, sc, *_ in stranger if sc < t)      # 陌生人被正确拒绝
    fa = sum(1 for _, sc, *_ in stranger if sc >= t)           # 误识：陌生人被判成某人
    fr = sum(1 for _, sc, *_ in known if sc < t)               # 拒识：本人被拒

    return (correct / len(samples) if samples else 0,
            fa / len(stranger) if stranger else 0,
            fr / len(known) if known else 0)


def main():
    bank = FaceBank()
    print("人脸库:", bank.stats())
    if not bank.all_names():
        print("人脸库为空，请先注册！")
        return
    if not TEST_DIR.exists():
        print(f"测试目录不存在：{TEST_DIR}\n请按 data/test/<标签>/xxx.jpg 组织测试照片")
        return

    samples = collect_samples(bank)
    if not samples:
        print("没有可用样本")
        return
    n_known = sum(1 for s in samples if s[0])
    print(f"\n共 {len(samples)} 张（本人 {n_known} / 陌生人 {len(samples) - n_known}）")

    g = [s[1] for s in samples if s[0]]       # 本人样本分数
    x = [s[1] for s in samples if not s[0]]   # 陌生人样本分数
    print(f"\n本人分数范围:  {min(g):.3f} ~ {max(g):.3f}")
    print(f"陌生人分数范围: {min(x):.3f} ~ {max(x):.3f}")
    print(f"决策间隔: ({max(x):.3f}, {min(g):.3f})，中点 = {(max(x)+min(g))/2:.3f}")

    print(f"{'阈值':>6} {'准确率':>8} {'误识率FAR':>10} {'拒识率FRR':>10}")

    rows = [metrics(samples, t) for t in THRESHOLDS]
    for t, (a, fa, fr) in zip(THRESHOLDS, rows):
        print(f"{t:>6.2f} {a:>8.3f} {fa:>10.3f} {fr:>10.3f}")

    # 推荐阈值 = 完美分界平台的中点，两侧留等量裕量
    gap_lo, gap_hi = max(x), min(g)
    if gap_lo < gap_hi:
        best_t = (gap_lo + gap_hi) / 2
        margin = (gap_hi - gap_lo) / 2
        note = (f"决策间隔 ({gap_lo:.3f}, {gap_hi:.3f})，"
                f"两侧裕量 ±{margin:.3f}")
    else:
        # 本人 / 陌生人分数有重叠，退而取最高准确率平台的中点
        max_acc = max(r[0] for r in rows)
        plateau = [t for t, r in zip(THRESHOLDS, rows) if r[0] == max_acc]
        best_t = (min(plateau) + max(plateau)) / 2
        note = f"本人/陌生人分数重叠，取最高准确率({max_acc:.3f})平台中点"

    ba, bfa, bfr = metrics(samples, best_t)
    print(f"\n>>> 推荐阈值：{best_t:.3f}（{note}）")
    print(f"    准确率 {ba:.3f}，FAR {bfa:.3f}，FRR {bfr:.3f}")

    accs, fars, frrs = map(list, zip(*rows))
    plt.figure(figsize=(8, 5))
    plt.plot(THRESHOLDS, accs, "o-", label="准确率")
    plt.plot(THRESHOLDS, fars, "s--", label="误识率 FAR")
    plt.plot(THRESHOLDS, frrs, "^--", label="拒识率 FRR")
    plt.axvline(best_t, color="gray", ls=":",
                label=f"推荐阈值 {best_t:.3f}")
    plt.xlabel("判定阈值（余弦相似度）"); plt.ylabel("比例")
    plt.title("阈值 - 性能曲线"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"曲线已保存：{OUT_PNG}")


if __name__ == "__main__":
    main()
