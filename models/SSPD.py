# -*- coding:utf-8 -*-
# @File   : SSPD.py
# @Time   : 2022/11/28 9:57
# @Author : Zhang Xinyu
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class Backbone_step1(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(3, 8, kernel_size=(3, 5, 5), padding=(1, 0, 0), bias=False)
        self.Conv_Layer2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 0, 0), bias=False)
        self.BN1 = nn.BatchNorm3d(8)
        self.BN2 = nn.BatchNorm3d(16)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Act = nn.Tanh()

    def forward(self, input):
        output = self.Act(self.BN1(self.Conv_Layer1(input)))
        output = self.Act(self.BN2(self.Conv_Layer2(output)))
        output = self.Avgpool1(output)
        return output


class Backbone_step2(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(16, 16, kernel_size=(3, 5, 5), padding=(1, 0, 0), bias=False)
        self.Conv_Layer2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 0, 0), bias=False)
        self.BN1 = nn.BatchNorm3d(16)
        self.BN2 = nn.BatchNorm3d(32)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Act = nn.Tanh()

    def forward(self, input):
        output = self.Act(self.BN1(self.Conv_Layer1(input)))
        output = self.Act(self.BN2(self.Conv_Layer2(output)))
        output = self.Avgpool1(output)
        return output


class Backbone_step3(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(32, 32, kernel_size=(3, 5, 5), padding=(1, 0, 0), bias=False)
        self.Conv_Layer2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 0, 0), bias=False)
        self.BN1 = nn.BatchNorm3d(32)
        self.BN2 = nn.BatchNorm3d(64)
        self.Avgpool1 = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.Act = nn.Tanh()

    def forward(self, input):
        output = self.Act(self.BN1(self.Conv_Layer1(input)))
        output = self.Act(self.BN2(self.Conv_Layer2(output)))
        output = self.Avgpool1(output)
        return output


class Backbone_step4(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_Layer1 = nn.Conv3d(64, 256, kernel_size=(3, 3, 3), padding=(1, 0, 0), bias=False)
        self.BN1 = nn.BatchNorm3d(256)
        self.Avgpool1 = nn.AvgPool3d((1, 8, 8), stride=(1, 1, 1))
        self.Act = nn.Tanh()
    def forward(self, input):
        output_bf_GAP = self.Act(self.BN1(self.Conv_Layer1(input)))
        output = self.Avgpool1(output_bf_GAP)
        return output_bf_GAP, output.squeeze(-1).squeeze(-1).squeeze(1)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(256, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.down = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn3 = nn.BatchNorm3d(64)
        self.GAP = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv4 = nn.Conv3d(64, 1, kernel_size=(1, 1, 1))
        self.act = nn.Tanh()

    def forward(self, input):
        output = self.act(self.bn1(self.conv1(input)))
        output = self.down(output)
        output = self.act(self.bn2(self.conv2(output)))
        output = F.interpolate(output, scale_factor=(2, 1, 1))
        output = self.act(self.bn3(self.conv3(output)))
        output = self.GAP(output)
        output = self.conv4(output)
        return output.squeeze(-1).squeeze(-1).squeeze(1)


class TokenizeConv(nn.Module):
    def __init__(self, input_ch, output_ch, k):
        super().__init__()
        self.conv = nn.Conv1d(input_ch, output_ch, kernel_size=k)
        self.norm = nn.LayerNorm(output_ch, eps=1e-6)

    def forward(self, input):
        output = rearrange(self.conv(input), 'b c t -> b t c')
        tokens = self.norm(output)
        return tokens


class MultiHeadedDotProductAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None

    def forward(self, x):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1)
        scores /= scores.size(-1)
        scores = self.drop(scores)
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h, scores


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class TSBlock(nn.Module):
    def __init__(self, dim, k, num_head, dropout=0):
        super().__init__()
        self.tokenizer = TokenizeConv(dim, dim, k)
        self.attn = MultiHeadedDotProductAttention(dim, num_head, dropout)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, x):
        t = self.tokenizer(x)
        a, scores = self.attn(t)
        h = self.act((self.proj(a)))
        x_down = F.interpolate(x, size=h.size()[1])
        r = x_down + rearrange(h, 'b t c -> b c t')
        return r, h


class SSPDL3(nn.Module):
    def __init__(self):
        super().__init__()
        self.BN = nn.BatchNorm3d(3)
        self.backbone_step1 = Backbone_step1()
        self.backbone_step2 = Backbone_step2()
        self.backbone_step3 = Backbone_step3()
        self.backbone_step4 = Backbone_step4()
        self.predictor = Predictor()
        self.ts_block1 = TSBlock(dim=256, k=9, num_head=2)
        self.ts_block2 = TSBlock(dim=256, k=7, num_head=2)
        self.ts_block3 = TSBlock(dim=256, k=5, num_head=2)
        self.apply(self.weight_init)

    def forward(self, input_):
        input_ = self.BN(input_)

        backbone_output_step1 = self.backbone_step1(input_)
        backbone_output_step2 = self.backbone_step2(backbone_output_step1)
        backbone_output_step3 = self.backbone_step3(backbone_output_step2)
        output_bf_gap, output_gap = self.backbone_step4(backbone_output_step3)

        rppg = self.predictor(output_bf_gap.detach())

        res1, output1 = self.ts_block1(output_gap)
        res2, output2 = self.ts_block2(res1)
        _, output3 = self.ts_block3(res2)

        return (output1, output2, output3), rppg

    @torch.no_grad()
    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class SSPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.BN = nn.BatchNorm3d(3)
        self.backbone_step1 = Backbone_step1()
        self.backbone_step2 = Backbone_step2()
        self.backbone_step3 = Backbone_step3()
        self.backbone_step4 = Backbone_step4()
        self.predictor = Predictor()
        self.apply(self.weight_init)

    def forward(self, input_):
        input_ = self.BN(input_)

        backbone_output_step1 = self.backbone_step1(input_)
        backbone_output_step2 = self.backbone_step2(backbone_output_step1)
        backbone_output_step3 = self.backbone_step3(backbone_output_step2)
        output_bf_gap, output_gap = self.backbone_step4(backbone_output_step3)

        rppg = self.predictor(output_bf_gap.detach())

        return rppg

    @torch.no_grad()
    def weight_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


