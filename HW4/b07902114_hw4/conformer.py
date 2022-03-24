"""
Refernece : https://github.com/lucidrains/conformer/blob/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class Switch(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def forward(self, x):
        x, x_ = x.chunk(2, dim=1)
        return x * x_.sigmoid()

class AttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate):
        super().__init__()
        self.MHSA = nn.MultiheadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, _ = self.MHSA(x, x, x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class DepthWiseCNN(nn.Module):
    def __init__(self, input_channel, output_channel, kernel):
        super().__init__()
        self.conv = nn.Conv1d(input_channel, output_channel, kernel, groups = input_channel)
        self.kernel = kernel
    def forward(self, x):
        x = F.pad(x, (self.kernel//2, self.kernel//2))
        x = self.conv(x)
        return x

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel, dropout_rate):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.LayerNorm(d_model),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(d_model, 4 * d_model, 1),
            GLU(),
            DepthWiseCNN(2 * d_model, 2 * d_model, kernel),
            nn.BatchNorm1d(2 * d_model),
            Switch(),
            nn.Conv1d(2 * d_model, d_model, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return self.cnn(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout_rate):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            Switch(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout_rate),
        )
    def forward(self, x):
        return self.fc(x)

class Conformer(nn.Module):
    def __init__(self, d_model, dim_feedforward, kernel, num_heads, dropout_rate):
        super().__init__()
        self.ff1 = FeedForward(d_model, dim_feedforward, dropout_rate)
        self.mhsa = AttentionModule(d_model, num_heads, dropout_rate)
        self.conv = ConvModule(d_model, kernel, dropout_rate)
        self.ff2 = FeedForward(d_model, dim_feedforward, dropout_rate)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = 0.5 * self.ff1(x) + x
        x = self.mhsa(x) + x
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        x = self.norm(x)
        return x

class SelfAttentivePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, 128)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, h):
        x = self.w1(h)
        x = self.relu(x)
        A = self.softmax(self.w2(x))
        h = h.transpose(1, 2)
        E = torch.matmul(h, A).squeeze(2)
        return E
        