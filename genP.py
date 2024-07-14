import sys
import os
import numpy as np
import torch
from torchvision.utils import save_image

sys.path.append('path_to_stylegan2_ada_pytorch')  # 替换为实际路径
import dnnlib
import legacy

# 设置设备为GPU或CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 加载预训练的StyleGAN模型
network_pkl = 'path_to_pretrained_model.pkl'  # 替换为实际路径
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)

# 生成随机潜在向量z
z = torch.randn([1, G.z_dim]).to(device)

# 生成图像
img = G(z, None)
img = (img + 1) * (255 / 2)
img = img.clamp(0, 255).to(torch.uint8)
img = img.permute(0, 2, 3, 1)

# 保存生成的图像
os.makedirs('generated_images', exist_ok=True)
save_image(img[0], 'generated_images/sample.png')
