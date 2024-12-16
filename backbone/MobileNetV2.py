
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

def Backbone_MobileNetV2(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    # 加载 torchvision 提供的 MobileNetV2 模型
    net = mobilenet_v2(pretrained=pretrained)

    # 提取 MobileNetV2 的特征部分
    features = list(net.features.children())

    # 根据特征图降维程度划分为不同阶段
    div_2 = nn.Sequential(*features[:2])  # Stage 1: 分辨率 / 2
    div_4 = nn.Sequential(*features[2:4])  # Stage 2: 分辨率 / 4
    div_8 = nn.Sequential(*features[4:7])  # Stage 3: 分辨率 / 8
    div_16 = nn.Sequential(*features[7:14])  # Stage 4: 分辨率 / 16
    div_32 = nn.Sequential(*features[14:])  # Stage 5: 分辨率 / 32

    return div_2, div_4, div_8, div_16, div_32

if __name__ == "__main__":
    # 构建骨干网络
    div_2, div_4, div_8, div_16, div_32 = Backbone_MobileNetV2(pretrained=True)

    # 打印模型结构
    print("Stage div_2:", div_2)
    print("Stage div_4:", div_4)
    print("Stage div_8:", div_8)
    print("Stage div_16:", div_16)
    print("Stage div_32:", div_32)

    # 输入测试
    x = torch.randn(16, 3, 352, 352)  # 模拟输入图像
    x2 = div_2(x)
    x4 = div_4(x2)
    x8 = div_8(x4)
    x16 = div_16(x8)
    x32 = div_32(x16)

    # 打印每阶段输出特征图的形状
    print("Feature shapes:")
    print("div_2 output shape:", x2.shape)
    print("div_4 output shape:", x4.shape)
    print("div_8 output shape:", x8.shape)
    print("div_16 output shape:", x16.shape)
    print("div_32 output shape:", x32.shape)