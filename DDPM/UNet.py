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
        i = torch.linspace(0, (d_model - 1) / 2, d_model // 2)

        pos = torch.unsqueeze(pos, 1)
        i = torch.unsqueeze(i, 0)

        pe_sin = torch.sin(pos / 10000 ** ((2 * i) / d_model))
        pe_cos = torch.cos(pos / 10000 ** ((2 * i) / d_model))

        pe = torch.stack((pe_sin, pe_cos), 2).reshape(seq_length, d_model)

        self.embedding = nn.Embedding(seq_length, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)


    def forward(self, t):
        return self.embedding(t)


