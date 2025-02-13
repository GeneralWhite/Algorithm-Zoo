import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, seq_length: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number
        assert d_model % 2 == 0

        pe = torch.zeros(seq_length, d_model)

        pos = torch.linspace(0, seq_length - 1, seq_length)
        i = torch.arange(0, d_model//2)

        # 位置编码的目标是为序列中每个位置生成一个d_model维的向量
        # 将pos和i扩展成(seq_length, 1), (1, d_model//2), 方便与(seq_length, d_model)对齐进行广播
        pos = torch.unsqueeze(pos, 1)
        i = torch.unsqueeze(i, 0)

        pe_sin = torch.sin(pos / 10000 ** ((2 * i) / d_model))
        pe_cos = torch.cos(pos / 10000 ** ((2 * i) / d_model))

        # pe_sin: (seq_length, d_model//2)
        # pe_cos: (seq_length, d_model//2)
        # torch.stack((pe_sin, pe_cos), 2): (seq_length, d_model//2, 2)
        pe = torch.stack((pe_sin, pe_cos), 2).reshape(seq_length, d_model)

        self.embedding = nn.Embedding(seq_length, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)


    def forward(self, t):
        return self.embedding(t)


class UnetBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, residual = False):
        super().__init__()
        # 层归一化, 不知道有什么用
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual

        # 当输入通道数和输出通道数相等时, 残差连接部分直接使用恒等映射
        # 否则使用1*1卷积调整通道数, 1*1卷积核可以改变通道数而不改变空间维度(高和宽)
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)

        # 残差连接, 将out和初始输入x相加
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)

        return out


def get_img_shape():
    # MNIST数据集 img_shape
    return (1, 28, 28)

# 输入通道和输出通道
# 输出通道数即为使用卷积核的数量
# 假设输入通道为3, 输出通道为32, 则使用32个卷积核对3个通道进行处理
# 每个卷积核独立对三个通道进行卷积, 每个卷积核生成3个特征图, 拼接起来即为1个输出通道, 32个卷积核独立处理
class Unet(nn.Module):
    def __init__(self, n_steps, channels = None, pe_dim = 10, residual = False):
        super().__init__()
        if channels is None:
            channels = [10, 20, 40, 80]

        C, H, W = get_img_shape()
        layers = len(channels)

        layers_H = [H]
        layers_W = [W]

        c_H = H
        c_W = W

        # 遍历到倒数第二层, 模拟下采样过程, c_H和c_W为每次下采样后特征图的H和W
        for _ in range(0, layers - 1):
            c_H //= 2
            c_W //= 2

            layers_H.append(c_H)
            layers_W.append(c_W)

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.pe_encoders = nn.ModuleList()
        self.pe_decoders = nn.ModuleList()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        pre_channel = C

        # Unet编码器部分, 每次下采样后通道数, 特征图的宽和高都会变化
        for channel, c_H, c_W in zip(channels[0:-1], layers_H[0:-1], layers_W[0:-1]):
            # 时间步编码头
            # 下面都是ModuleList套Sequential结构, 每个Sequential作为ModuleList中的一个元素
            self.pe_encoders.append(
                nn.Sequential(
                    nn.Linear(pe_dim, pre_channel),
                    nn.ReLU(),
                    nn.Linear(pre_channel, pre_channel)
                )
            )

            self.encoders.append(
                nn.Sequential(
                    UnetBlock((pre_channel, c_H, c_W),
                              pre_channel,
                              channel,
                              residual = residual),
                    UnetBlock((channel, c_H, c_W),
                              channel,
                              channel,
                              residual = residual),
                )
            )

            # 下采样: 特征图尺度减半(2*2卷积核, 步长为2)
            self.downs.append(
                nn.Conv2d(channel, channel, 2, 2)
            )

            pre_channel = channel

        # 中间层处理
        self.pe_mid = nn.Linear(pe_dim, pre_channel)
        channel = channels[-1]

        self.mid = nn.Sequential(
            UnetBlock((pre_channel, layers_H[-1], layers_W[-1]),
                      pre_channel,
                      channel,
                      residual = residual),
            UnetBlock((channel, layers_H[-1], layers_W[-1]),
                      channel,
                      channel,
                      residual = residual)
        )
        pre_channel = channel

        # channels[-2::-1]: 列表[起始:结束:步长]
        # 从倒数第二个元素开始, 结束位置为表头, 步长为-1
        for channel, c_H, c_W in zip(channels[-2::-1], layers_H[-2::-1], layers_W[-2::-1]):
            self.pe_decoders.append(
                nn.Linear(pe_dim, pre_channel)
            )

            self.ups.append(
                # 反卷积上采样
                nn.ConvTranspose2d(pre_channel, channel, 2, 2)
            )

            self.decoders.append(
                # 这里输入是channel*2, 因为在Unet解码器部分要跳跃连接, 要将编码器对应部分拼接后输入到decoder
                # 这里输入是channel*2而不是pre_channel*2, 因为解码器中先进行上采样再输入到decoder中
                nn.Sequential(
                    UnetBlock((channel * 2, c_H, c_W),
                              channel * 2,
                              channel,
                              residual = residual),
                    UnetBlock((channel, c_H, c_W),
                              channel,
                              channel,
                              residual = residual)))

            pre_channel = channel

            # 输出投影图, 保持输入尺寸(3*3卷积核, 步长1, 填充1不会导致特征图变化)
            self.conv_out = nn.Conv2d(pre_channel, C, 3, 1, 1)


    def forwrd(self, x, t):
        n = t.shape[0]
        t = self.pe(t)

        encoder_outs = []

        for pe_encoder, encoder, down in zip(self.pe_encoders, self.encoders, self.downs):
            # pe_encoder(t)的维度是 (batch, channel), 因后续要与x相加, 故要reshape成(batch, channel, 1, 1)方便广播
            # x的维度是 (batch, channel, H, W)
            pe = pe_encoder(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            # 保存用于解码器跳跃连接
            encoder_outs.append(x)

            x = down(x)

        pe = self.pe_mid(t).reshape(n, -1, 1, 1)
        x = self.mid(x + pe)

        for pe_decoder, decoder, up, encoder_out in zip(self.pe_decoders, self.decoders, self.ups, encoder_outs):
            pe = pe_decoder(t).reshape(n, -1, 1, 1)
            # 编码时先编码再下采样, 解码时先上采样再解码, 顺序相反
            x = up(x)

            # 上采样后, 特征图的尺寸可能和编码器对应的尺寸出现差异, 为了跳跃连接, 需要对尺寸进行填充校准
            pad_y = encoder_out.shape[2] - x.shape[2]  # H
            pad_x = encoder_out.shape[3] - x.shape[3]  # W

            # 对称填充, 先填充W(左右方向), 在填充H(上下方向)
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2))

            # 跳跃连接, 在通道维度连接
            x = torch.concat((encoder_out, x), dim = 1)

            x = decoder(x + pe)

        x = self.conv_out(x)
        return x


unet_1_cfg = {
    'channels': [10, 20, 40, 80],
    'pe_dim': 128
}


unet_res_cfg = {
    'channels': [10, 20, 40, 80],
    'pe_dim': 128,
    'residual': True
}


def build_network(config: dict, n_steps):
    ## **用于解包字典为关键字参数
    network = Unet(n_steps, **config)

    return network