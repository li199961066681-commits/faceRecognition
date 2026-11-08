"""人脸检测与对齐：SCRFD 检测人脸框+五关键点，按五点裁剪 112x112 标准人脸。"""
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from insightface.utils.face_align import norm_crop


def imread_cn(path):
    """支持中文路径的图片读取"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class FaceDetector:
    def __init__(self, det_size=(640, 640), det_thresh=0.5):
        self.app = FaceAnalysis(
            name="buffalo_m",
            allowed_modules=["detection"],      # 只用检测模块，识别特征由 extractor 负责
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=-1, det_size=det_size, det_thresh=det_thresh)

    def detect(self, image_bgr):
        """返回该图所有检测到的人脸（insightface Face 对象列表）"""
        return self.app.get(image_bgr)

    def get_main_face(self, bgr, max_side=1600):
        """检测画面中最大的人脸，内置多尺度重试（解决大特写脸漏检）。
        返回 (bbox[x1,y1,x2,y2,score]（原图坐标）, aligned 112x112)；未检出返回 (None, None)。"""
        img = bgr
        H, W = img.shape[:2]
        k0 = min(1.0, max_side / max(H, W))  # 超大图先统一缩档，兼顾速度
        if k0 < 1.0:
            img = cv2.resize(img, (int(W * k0), int(H * k0)))

        for fx in (1.0, 0.6, 0.4, 0.25):
            test = img if fx == 1.0 else cv2.resize(img, None, fx=fx, fy=fx)
            faces = self.app.get(test)
            if faces:
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                k = k0 * fx  # test图坐标 → 原图坐标 的换算
                x1, y1, x2, y2 = f.bbox[:4] / k
                bbox = np.array([x1, y1, x2, y2, float(f.det_score)])
                aligned = norm_crop(bgr, f.kps / k)  # 关键点还原回原图，在原图上对齐（保清晰度）
                return bbox, aligned
        return None, None

