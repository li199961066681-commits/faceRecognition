# -*- coding: utf-8 -*-
"""app.py —— 人脸识别系统界面
运行：在项目根目录的终端里执行  streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from core import FaceBank, recognize

st.set_page_config(page_title="人脸识别系统", page_icon="🧑‍💼", layout="wide")


@st.cache_resource
def load_bank():
    """模型与人脸库只加载一次（切换页面/上传文件不会重复加载）。"""
    return FaceBank()


bank = load_bank()

# ---------- 侧边栏 ----------
page = st.sidebar.radio("功能选择", ["① 人员注册", "② 人脸识别"])
threshold = st.sidebar.slider("判定阈值：最高相似度 ≥ 阈值 判为本人", 0.30, 0.70, 0.50, 0.01)

st.sidebar.subheader("人脸库状态")
names = bank.all_names()
if names:
    for n, c in bank.stats().items():
        st.sidebar.write(f"- **{n}**：{c} 张")
else:
    st.sidebar.write("（空，请先注册）")


# ---------- 工具函数 ----------
def decode_image(uploaded):
    """Streamlit 上传文件 → BGR 图（字节解码，不经过路径，天然支持中文文件名）。"""
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def imread_safe(path):
    """中文路径安全读图（cv2.imread 会静默失败）。"""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def draw_result(bgr, bbox, text, ok=True):
    """画框 + 中文标签（cv2.putText 不支持中文，用 PIL 画）。"""
    img = bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    color = (0, 200, 0) if ok else (220, 0, 0)            # RGB 色给 PIL 用
    cv2.rectangle(img, (x1, y1), (x2, y2), (color[2], color[1], color[0]), 3)

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
    except OSError:
        font = ImageFont.load_default()
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    ty = max(0, y1 - (b - t) - 14)
    draw.rectangle([x1, ty, x1 + (r - l) + 12, ty + (b - t) + 12], fill=color)
    draw.text((x1 + 6, ty + 6), text, font=font, fill=(255, 255, 255))
    return np.array(pil)


# ================= 页面一：注册 =================
if page == "① 人员注册":
    st.header("① 人员注册")
    st.caption("每人上传 3~5 张不同角度/表情的照片；构图参考已验证成功的那类。")

    name = st.text_input("姓名")
    files = st.file_uploader("选择照片", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)

    if st.button("开始注册", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("请先填写姓名")
        elif not files:
            st.warning("请先选择照片")
        else:
            ok_list, fail_list = [], []
            for f in files:
                img = decode_image(f)
                if img is None:
                    fail_list.append((f.name, "图片解码失败"))
                    continue
                r = bank.register(name, img)
                (ok_list.append(r) if r["ok"] else fail_list.append((f.name, r["msg"])))

            if ok_list:
                st.success(f"「{name}」本次成功 {len(ok_list)} 张，累计 {ok_list[0]['n']} 张")
                cols = st.columns(min(5, len(ok_list)))
                for c, r in zip(cols, ok_list):
                    if r["photo"]:
                        face = imread_safe(r["photo"])
                        if face is not None:
                            c.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), width=120)
            for fname, msg in fail_list:
                st.error(f"{fname}：{msg}")

    st.divider()
    st.subheader("删除人员")
    if names:
        del_name = st.selectbox("选择要删除的人员", names)
        if st.button("删除该人员"):
            bank.remove(del_name)
            st.success(f"已删除 {del_name}")

# ================= 页面二：识别 =================
else:
    st.header("② 人脸识别")
    f = st.file_uploader("上传查询照片", type=["jpg", "jpeg", "png"])

    if st.button("开始识别", type="primary", use_container_width=True):
        if f is None:
            st.warning("请先上传照片")
        else:
            img = decode_image(f)
            res = recognize(img, bank, threshold)

            if res["status"] == "no_face":
                st.error(res["msg"])
            else:
                known = res["status"] == "known"
                label = res["name"] if known else "陌生人"
                st.image(draw_result(img, res["bbox"],
                                     f"{label}  {res['score']:.3f}", ok=known))
                c1, c2 = st.columns(2)
                c1.metric("判定结果", label)
                c2.metric("最高相似度", f"{res['score']:.4f}",
                          f"阈值 {threshold:.2f}", delta_color="off")
                st.subheader("Top-3 候选")
                st.table({"姓名": [n for n, _ in res["top3"]],
                          "相似度": [s for _, s in res["top3"]]})
