# faceRecognition —— 人脸识别系统

基于 **InsightFace (SCRFD)** 人脸检测 + **ArcFace (ResNet18)** 特征提取的人脸识别系统，提供 Streamlit 可视化界面，支持人员注册、1:1 / 1:N 识别、阈值调优。

## ✨ 功能

- 🧑 **人员注册**：上传 3~5 张不同角度/表情的照片，自动检测人脸、对齐、提取 512 维特征并存库
- 🔍 **人脸识别**：上传查询照片，返回 Top-3 候选姓名与相似度，高于阈值判为本人，否则判为陌生人
- 📊 **阈值调优**：`threshold_test.py` 在测试集上扫描阈值，输出准确率 / FAR / FRR 曲线，自动推荐"决策间隔中点"作为阈值
- 🛡️ **best-of-k 匹配**：每人保留多张照片的特征向量，识别时取该人所有特征中的最高相似度（不做平均，保留角度多样性）
- 🌐 **中文路径支持**：`imread_cn` 用 `np.fromfile + cv2.imdecode` 读写图片，避免 OpenCV 在中文路径下静默失败

## 🧠 技术栈

| 环节 | 方案 |
|---|---|
| 人脸检测 + 关键点 | InsightFace **SCRFD**（buffalo_m / det_2.5g，CPU onnxruntime） |
| 人脸对齐 | 五关键点 `norm_crop` → 112×112 标准脸 |
| 特征提取 | **ArcFace + ResNet18 (IR-SE)**，输入 128×128 灰度，输出 512 维 L2 归一化向量 |
| 相似度 | 余弦相似度（特征已归一化，内积即余弦） |
| 界面 | Streamlit |

权重来源：[GFPGAN 官方 arcface_resnet18.pth](https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth)，首次运行自动下载到 `weights/`。

## 📁 项目结构

```
faceRecognition/
├── app.py                  # Streamlit 主界面（注册 / 识别 两个页面）
├── threshold_test.py       # 阈值扫描与推荐脚本
├── requirements.txt
├── core/
│   ├── __init__.py
│   ├── detector.py         # FaceDetector：SCRFD 检测 + 多尺度重试 + norm_crop 对齐
│   ├── extractor.py        # FaceExtractor：ArcFace 特征提取（自动下载权重）
│   ├── arcface_r18.py      # ResNet18(IR-SE) 网络结构定义
│   ├── facebank.py         # FaceBank：注册 / 搜索 / 删除 / 持久化
│   └── recognizer.py       # recognize()：检测→提特征→查库→阈值判定
├── weights/                # 模型权重（.gitignore 忽略，自动下载）
└── data/                   # 注册照片 + facebank.json（.gitignore 忽略，含敏感数据不上传）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> 首次运行会自动下载 ArcFace 权重（~98MB）到 `weights/arcface_resnet18.pth`。如遇网络问题，可手动下载放到该路径。

### 2. 启动界面

```bash
streamlit run app.py
```

浏览器打开后：
- **① 人员注册**：输入姓名 → 选择照片（支持多选）→ 开始注册
- **② 人脸识别**：上传查询照片 → 开始识别，侧栏可调节判定阈值

### 3. 阈值调优（可选）

将测试照片按 `data/test/<标签>/*.jpg` 组织（标签=已注册人名为本人样本，其他为陌生人样本），运行：

```bash
python threshold_test.py
```

脚本会输出本人/陌生人分数范围、决策间隔、各阈值下的准确率/FAR/FRR，并推荐一个两侧裕量相等的阈值，保存 `threshold_curve.png`。

## ⚙️ 判定规则

```
score = max(库中该人所有特征与查询特征的余弦相似度)
score >= threshold  →  本人 (known)
score <  threshold  →  陌生人 (stranger)
未检测到人脸         →  no_face
```

## 📝 注意事项

- **检测阈值**：`FaceDetector(det_thresh=0.5)`。极端贴脸特写（人脸占画面 >70%）可能因置信度不足漏检，可考虑降低阈值或检测前给画面补边距。
- **注册构图**：建议每人 3~5 张正脸、不同角度/表情、光线充足的照片。
- **数据安全**：`data/` 目录含注册人脸照片与特征向量，已在 `.gitignore` 中排除，**请勿上传到公开仓库**。
