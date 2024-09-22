# -*- coding:utf-8 -*-
# @File   : SSPDLoss.py
# @Time   : 2023/2/26 16:02
# @Author : Zhang Xinyu
import torch
import torch.nn as nn
from utils.fft_package import Turn_map_into_waves, FFT_Batch

class SSPDLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_distill_pyramid = Loss_Distill_Pyramid()
        self.loss_distill_rPPG = Loss_Distill_rPPG(args.GPU_id)
        self.loss_reg = Loss_Reg()
        self.loss_freq = Loss_Freq(args.GPU_id)
        self.args = args

    def forward(self, target_output, online_output):
        pyramid_online, pyramid_target = online_output[0], target_output[0]
        rPPG_online, rPPG_target = online_output[1], target_output[1]
        loss_pyramid = 0
        loss_reg = 0
        loss_freq = 0
        target_SSM_return = []
        online_SSM_return = []
        for i in range(len(pyramid_online)):
            _target_output = pyramid_target[i].detach()
            _online_output = pyramid_online[i]
            target_SSM = self_cos_similarity_calc(_target_output.unsqueeze(1)).squeeze(1)
            online_SSM = self_cos_similarity_calc(_online_output.unsqueeze(1)).squeeze(1)
            target_SSM_return.append(target_SSM.tolist())
            online_SSM_return.append(online_SSM.tolist())
            loss_pyramid += self.loss_distill_pyramid(online_SSM, target_SSM)
            loss_reg += self.loss_reg(online_SSM) / len(pyramid_online)
            loss_freq += self.loss_freq(online_SSM) / len(pyramid_online)
        loss_rPPG = self.loss_distill_rPPG(rPPG_online, rPPG_target.detach()) * 0.5
        loss_pyramid = loss_pyramid * 0.5
        loss_reg = loss_reg * 0.5
        loss_freq = loss_freq * 0.2
        total_loss = loss_rPPG + loss_pyramid + loss_reg + loss_freq
        return total_loss, loss_pyramid.item(), loss_rPPG.item(), loss_reg.item(), loss_freq.item(), target_SSM_return, online_SSM_return


class Loss_Distill_Pyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.L2Loss = nn.MSELoss()
        self.PearsonLoss = NPearsonLoss()
        self.turner = Turn_map_into_waves()

    def forward(self, online_output, target_output):
        target_output = target_output.detach()
        wave_online = self.turner(online_output)
        wave_target = self.turner(target_output)
        total_loss = self.L2Loss(online_output, target_output) + self.L2Loss(wave_online, wave_target) + \
                     self.PearsonLoss(online_output, target_output) + self.PearsonLoss(wave_online, wave_target)
        return total_loss


class Loss_Distill_rPPG(nn.Module):
    def __init__(self, GPU_id):
        super().__init__()
        self.PearsonLoss = NPearsonLoss()
        self.fft = FFT_Batch(GPU_id)
        self.L2loss = nn.MSELoss(reduction='sum')

    def forward(self, online_output, target_output):
        target_output = target_output.detach()
        temporal_loss = self.PearsonLoss(online_output, target_output)
        lf = int(0.5 * online_output.shape[-1] / 30.0)
        hf = int(3 * online_output.shape[-1] / 30.0)
        vhf = int(15 * online_output.shape[-1] / 30.0)
        spectrum_online = self.fft(online_output)[:, :vhf]
        spectrum_target = self.fft(target_output)[:, :vhf]
        spectrum_online = spectrum_online / torch.max(spectrum_online, dim=-1, keepdim=True)[0]
        spectrum_target = spectrum_target / torch.max(spectrum_target, dim=-1, keepdim=True)[0]
        partition = torch.sum(spectrum_online, dim=-1) / torch.sum(spectrum_online[:, lf:hf], dim=-1) - 1
        freq_loss = self.PearsonLoss(spectrum_online, spectrum_target) + partition.mean()
        total_loss = temporal_loss + freq_loss
        return total_loss


class NPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target):
        loss = torch.mean(1 - pearson_corr_calc(predict, target))
        return loss


class Loss_Freq(nn.Module):
    def __init__(self, GPU_id):
        super().__init__()
        self.turner = Turn_map_into_waves()
        self.fft = FFT_Batch(GPU_id)

    def forward(self, attns):
        waves = self.turner(attns)
        wave_psd = self.fft(waves)
        lf = int(0.5 * wave_psd.shape[-1] / 30.0)
        hf = int(3 * wave_psd.shape[-1] / 30.0)
        wave_psd = wave_psd[:, :wave_psd.shape[-1]//2]
        wave_psd = wave_psd / torch.max(wave_psd, dim=-1, keepdim=True)[0]
        loss = torch.sum(wave_psd, dim=-1) / torch.sum(wave_psd[:, lf:hf], dim=-1) - 1
        return loss.mean()


class Loss_Reg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, SSMap):
        loss_batch = []
        for i in range(SSMap.shape[0]):
            loss_batch.append(self.std_regular(SSMap[i]))
        loss = torch.mean(torch.stack(loss_batch))
        return loss

    @staticmethod
    def std_regular(input):
        diag_list = [[] for _ in range(input.shape[0]-1)]
        for i in range(input.shape[0]-1):
            for j in range(i+1, input.shape[1]):
                diag_list[j-i-1].append(input[i, j])
        result = []
        for i in range(len(diag_list)-1):
            result.append(torch.std(torch.stack(diag_list[i])*len(diag_list[i])/20))
        loss = torch.mean(torch.stack(result))
        return loss


def pearson_corr_calc(x, y, batch_first=True, beta=1e-7):
    if batch_first:
        dim = -1
    else:
        dim = 0
    centered_x = x - torch.mean(x, dim=dim, keepdim=True)
    centered_y = y - torch.mean(y, dim=dim, keepdim=True)
    covariance = torch.sum(centered_x * centered_y, dim=dim, keepdim=True)
    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)
    x_std = torch.std(x, dim=dim, keepdim=True)
    y_std = torch.std(y, dim=dim, keepdim=True)
    corr = bessel_corrected_covariance / (x_std * y_std + beta)
    return corr.clamp_(min=-1.0, max=1.0)


def self_cos_similarity_calc(projection_space):
    projection_space = projection_space.squeeze(1)
    Numerator = projection_space @ projection_space.permute(0, 2, 1)
    Denominator = projection_space.mul(projection_space)
    Denominator = torch.sqrt(torch.sum(Denominator, dim=2, keepdim=True))
    Denominator = Denominator @ Denominator.permute(0, 2, 1)
    result = Numerator / Denominator
    return result
