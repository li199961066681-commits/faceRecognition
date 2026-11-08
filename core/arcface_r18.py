"""ArcFace 风格的 ResNet18 (IR-SE) 骨干网络。

权重来源：GFPGAN 官方发布的 arcface_resnet18.pth
https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/arcface_resnet18.pth
该模型用 ArcFace 损失在人脸上预训练，输入 128x128 灰度对齐人脸，输出 512 维特征。
结构定义与 GFPGAN/gfpgan/archs/arcface_arch.py 一致（保证能正确加载权重）。
"""
import torch
import torch.nn as nn


def conv3x3(inplanes, outplanes, stride=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class IRBlock(nn.Module):
    """改进残差块(IR)：BN -> Conv -> BN -> PReLU -> Conv -> (SE) -> 残差相加"""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.prelu(out)


class SEBlock(nn.Module):
    """通道注意力：给每个通道一个 0~1 的权重，突出重要通道"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.PReLU(),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNetArcFace(nn.Module):
    """ResNet18 形状（layers=[2,2,2,2]）的 ArcFace 骨干，输入 1 通道 128x128"""

    def __init__(self, block, layers, use_se=True):
        super().__init__()
        if block == "IRBlock":
            block = IRBlock
        self.inplanes = 64
        self.use_se = use_se
        # 注意 conv1 是 1 通道（灰度输入）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, 512)   # 128x128 输入 -> 8x8 特征图
        self.bn5 = nn.BatchNorm1d(512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, use_se=self.use_se)]
        self.inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.prelu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        return self.bn5(x)          # 512 维特征


def load_arcface_r18(weights_path, device="cpu"):
    """读取权重文件并构建网络（自动判断是否带 SE 模块）"""
    try:
        obj = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:               # 旧版 torch 没有 weights_only 参数
        obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, dict):       # 兼容 basicsr 保存格式
        for key in ("params_ema", "params", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in obj.items()}

    use_se = any(".se." in k for k in sd.keys())
    model = ResNetArcFace("IRBlock", (2, 2, 2, 2), use_se=use_se)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 2:            # 仅允许个别统计量丢失
        raise RuntimeError(f"权重加载异常 missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()
