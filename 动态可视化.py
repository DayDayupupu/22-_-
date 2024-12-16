import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyrealsense2 as rs
from collections import OrderedDict

from options import opt
from model.CIRNet_Res50 import CIRNet_R50
from model.CIRNet_vgg16 import CIRNet_V16

# 设置CUDA设备
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# 加载模型
print("加载模型...")
if opt.backbone == 'R50':
    model = CIRNet_R50()
else:
    model = CIRNet_V16()
print('使用骨干网络:', opt.backbone)

# 检查并使用多个GPU
gpu_num = torch.cuda.device_count()
if gpu_num == 1:
    print("使用单个GPU -", opt.gpu_id)
elif gpu_num > 1:
    print("使用多个GPU -", opt.gpu_id)


# 加载预训练模型参数并去掉 "module." 前缀
state_dict = torch.load('CIRNet_cpts/' + opt.test_model, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k  # 去掉 "module." 前缀
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)

model.cuda()
model.eval()

# 配置 RealSense 管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动 RealSense 摄像头
pipeline.start(config)
print("开始实时检测...")

try:
    while True:
        # 获取帧
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            print("无法读取帧，继续...")
            continue

        # 将帧转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 将帧转换为 Tensor 并传入 CUDA
        image_tensor = torch.tensor(color_image).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        depth_tensor = torch.tensor(depth_image).unsqueeze(0).unsqueeze(0).float().cuda() / 1000.0  # 假设深度以毫米为单位

        # 进行模型预测
        _, _, prediction = model(image_tensor, depth_tensor)
        prediction = F.interpolate(prediction, size=color_image.shape[:2], mode='bilinear', align_corners=False)
        prediction = prediction.sigmoid().data.cpu().numpy().squeeze()

        # 处理预测结果并显示
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        prediction = (prediction * 255).astype(np.uint8)
        prediction_colormap = cv2.applyColorMap(prediction, cv2.COLORMAP_JET)

        # 显示 RGB 图像和预测结果
        combined_image = np.hstack((color_image, prediction_colormap))
        cv2.imshow("Real-Time Detection", combined_image)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 结束并释放资源
    pipeline.stop()
    cv2.destroyAllWindows()
    print("实时检测结束。")
