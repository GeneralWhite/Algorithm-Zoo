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

