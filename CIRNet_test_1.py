import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score

from dataLoader import test_dataset
from model.CIRNet_Res50 import CIRNet_R50
from model.CIRNet_vgg16 import CIRNet_V16
from model.CIRNet_MobileNetV2 import CIRNet_MoV2
from options import opt

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# 加载模型
print('load model...')
# if opt.backbone == 'R50':
#     model = CIRNet_R50()
# else:
#     model = CIRNet_V16()
model = CIRNet_V16()
print('Use backbone ' + opt.backbone)

gpu_num = torch.cuda.device_count()

# 多 GPU 支持
if gpu_num == 1:
    print("Use Single GPU-", opt.gpu_id)
elif gpu_num > 1:
    print("Use multiple GPUs-", opt.gpu_id)
    model = torch.nn.DataParallel(model)

# 去除 "module." 前缀
from collections import OrderedDict

state_dict = torch.load('CIRNet_cpts/' + opt.test_model, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k  # 去掉 "module." 前缀
    new_state_dict[name] = v

# 加载 state_dict
model.load_state_dict(new_state_dict, strict=False)
model.cuda()
model.eval()

# test_datasets = ['SIP', 'DUT', 'NJU2K', 'STERE', 'NLPR', 'LFSD']
test_datasets = ['SIP', 'DUT', 'NJU2K', 'STERE', 'NLPR', 'LFSD']

dataset_path = opt.test_path

for dataset in test_datasets:
    print("Testing {} ...".format(dataset))
    save_path = 'test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    depth_root = dataset_path + dataset + '/depth/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, depth_root, gt_root, opt.testsize)

    mae_sum = 0
    ssim_sum = 0
    max_f_measure = []

    for i in range(test_loader.size):
        image_s, depth_s, gt_s, name = test_loader.load_data()
        name = name.split('/')[-1]

        gt = np.asarray(gt_s, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image_s.cuda()
        depth = depth_s.cuda()

        _, _, pre = model(image, depth)
        pre_s = F.interpolate(pre, size=gt.shape, mode='bilinear', align_corners=False)
        pre = pre_s.sigmoid().data.cpu().numpy().squeeze()
        pre = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)

        # 计算 MAE
        mae = np.mean(np.abs(pre - gt))
        mae_sum += mae

        # 计算 S Measure (结构相似性)
        ssim_value = ssim(gt, pre, data_range=pre.max() - pre.min())
        ssim_sum += ssim_value

        # 计算最大 F 测量 (Max F Measure)
        f_measures = []
        for threshold in np.linspace(0, 1, 5):  # 不同阈值下的 F 值
            pre_binary = (pre >= threshold).astype(int)
            gt_binary = (gt >= 0.5).astype(int)
            f_measure = f1_score(gt_binary.flatten(), pre_binary.flatten())
            f_measures.append(f_measure)
        max_f_measure.append(max(f_measures))

        # cv2.imwrite(save_path + name, pre * 255)

    avg_mae = mae_sum / test_loader.size
    avg_ssim = ssim_sum / test_loader.size
    avg_max_f = np.mean(max_f_measure)

    print(f"Dataset: {dataset} testing completed.")
    print(f"MAE: {avg_mae:.4f}, S Measure (SSIM): {avg_ssim:.4f}, Max F Measure: {avg_max_f:.4f}")

print("Test Done!")
