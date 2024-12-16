import os
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from options import opt
from model.CIRNet_Res50 import CIRNet_R50
from model.CIRNet_vgg16 import CIRNet_V16
from model.CIRNet_MobileNetV2 import CIRNet_MoV2
# 设置 CUDA 可见设备
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# 加载模型
print('加载模型...')
model = CIRNet_R50()
# model = CIRNet_V16
print(f'使用骨干网 {opt.backbone}')

# 检查并设置 GPU 数量
gpu_num = torch.cuda.device_count()
if gpu_num == 1:
    print(f"使用单 GPU - {opt.gpu_id}")
elif gpu_num > 1:
    print(f"使用多 GPU - {opt.gpu_id}")
    model = torch.nn.DataParallel(model)

# 加载模型权重
state_dict = torch.load(f'CIRNet_cpts/{opt.test_model}', map_location='cpu')
new_state_dict = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())
model.load_state_dict(new_state_dict, strict=False)

model.cuda()
model.eval()

# 设置样本路径
dataset_path = 'data/data_solo'  # 样本路径
image_path = os.path.join(dataset_path, '000008_常见.jpg')
depth_path = os.path.join(dataset_path, '000008_常见_depth.jpg')
gt_path = os.path.join(dataset_path, '0001h.jpg')  # 可选
save_path = 'test_maps/your_sample/'  # 结果保存路径

# 创建保存路径
os.makedirs(save_path, exist_ok=True)

# 定义图像转换
img_transform = transforms.Compose([
    transforms.Resize((352, 352)),  # 适合尺寸
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

depth_transform = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

# 定义图像加载函数
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

# 加载并转换图像
image = img_transform(rgb_loader(image_path)).unsqueeze(0).cuda()
depth = depth_transform(binary_loader(depth_path)).unsqueeze(0).cuda()

# 推理
with torch.no_grad():
    rgb, depth, pre = model(image, depth)
    pre_s = F.interpolate(pre, size=image.shape[2:], mode='bilinear', align_corners=False)
    pre = pre_s.sigmoid().cpu().numpy().squeeze()
    pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)


# 保存 RGB 和深度图像
save_name_rgb = os.path.basename(image_path)
save_name_rgb = f"CIRNet_epoch_65_rgb_{save_name_rgb}"  # 添加前缀
cv2.imwrite(os.path.join(save_path, save_name_rgb), rgb.squeeze().cpu().numpy() * 255)

save_name_depth = os.path.basename(image_path)
save_name_depth = f"CIRNet_epoch_65_depth_{save_name_depth}"  # 添加前缀
cv2.imwrite(os.path.join(save_path, save_name_depth), depth.squeeze().cpu().numpy() * 255)

# 保存预测结果
save_name_pred = os.path.basename(image_path)
save_name_pred = f"CIRNet_epoch_65_pred_{save_name_pred}"  # 添加前缀
cv2.imwrite(os.path.join(save_path, save_name_pred), pre * 255)

print("测试完成！")
