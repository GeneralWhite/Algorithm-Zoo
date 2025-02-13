import time
import einops
import torch.nn as nn
import torch.optim
import cv2
import numpy as np

from ddpm import DDPM
from dataset import get_dataloader
from UNet import build_network, unet_1_cfg, unet_res_cfg, get_img_shape

batch_size = 512
epochs = 100

def train(ddpm: DDPM, net, device = 'cuda', ckpt_path = './DDPM/model.pth'):
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for epoch in range(epochs):
        total_loss = 0

        for x, _ in dataloader:
            current_batch = x.shape[0]
            x = x.to(device)
            # 生成[0, n_steps)的随机时间步t
            # 这里生成的是1维张量, 虽然将(current_batch,)改为current_batch后结果没区别, 但使用(current_batch,)更规范
            # torch.randint参数为low, high, size; size要指定返回张量的形状(tuple 或 list)
            t = torch.randint(0, n_steps, (current_batch,)).to(device)
            # 生成噪声
            eps = torch.randn_like(x).to(device)
            # DDPM前向过程
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch, 1))
            loss = loss_fn(eps_theta, eps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {epoch} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample_imgs(ddpm, net, output_path, n_sample = 81, device = 'cuda', simple_var = True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        # * 解包元组
        # get_img_shape返回(1, 28, 28)
        # 解包后shape = (n_sample, 1, 28, 28)
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = ddpm.sample_backward(shape, net, device = device, simple_var = simple_var).detach().cpu()
        # 对图像进行恢复, 输入数据开始被归一化到[-1, 1], 可视化时需要恢复到初始的[0, 1]
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        # b1：网格的行数（由 n_sample 的平方根决定）
        # b2：网格的列数（隐含为 n_sample // b1）
        # 最终效果：将 n_sample 张图排列成 b1 x b2 的网格大图，便于可视化保存。
        imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1 = int(n_sample ** 0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite(output_path, imgs)


configs = [unet_1_cfg, unet_res_cfg]


if __name__ == '__main__':
    n_steps = 1000
    config_id = 1
    device = 'cuda'
    model_path = './model_unet_res.pth'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device = device, n_steps = n_steps)

    train(ddpm, net, device = device, ckpt_path = model_path)
    net.load_state_dict(torch.load(model_path))
    sample_imgs(ddpm, net, './diffusion.jpg')



