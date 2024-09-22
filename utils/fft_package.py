# Copyright ©2022 Zhang Xinyu, Sun weiyu and Chen ying. All Rights Reserved.
import torch
from torch import nn

class FFT_MODULE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        batch_list = torch.split(input, 1, 0)
        result_list = []
        for i in range(len(batch_list)):
            attn_zeros = torch.zeros(batch_list[i].shape[1:]).unsqueeze(-1).cuda()
            attn_real_and_image = torch.cat([batch_list[i].squeeze().unsqueeze(-1), attn_zeros], -1)
            try:
                result = torch.fft(attn_real_and_image, 2)
            except:
                result = torch.fft.fft(attn_real_and_image, 2)
            final_result = torch.norm(result, p=2, dim=2).unsqueeze(0)
            result_list.append(final_result)

        result = torch.cat(result_list, 0)
        return result


class FFT_Batch(nn.Module):
    def __init__(self, GPU_id, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.GPU_id = GPU_id

    def forward(self, input):
        if self.use_cuda:
            attn_zeros = torch.zeros(input.shape).unsqueeze(-1).to(torch.device(self.GPU_id))
        else:
            attn_zeros = torch.zeros(input.shape).unsqueeze(-1)
        attn_real_and_image = torch.cat([input.unsqueeze(-1), attn_zeros], -1)
        # Pytorch version compatible.
        try:
            result = torch.fft(attn_real_and_image, 1)
        except:
            result = torch.fft.fft(input).unsqueeze(-1)
        PSD = torch.norm(result, p=2, dim=-1)
        return PSD


class FFT_MODULE_1d(nn.Module):
    def __init__(self, GPU_id, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.GPU_id = GPU_id

    def forward(self, input):
        pass

    def solo_fft_1d_calc(self, input):
        if self.use_cuda:
            attn_zeros = torch.zeros(input.shape).unsqueeze(-1).to(torch.device(self.GPU_id))
        else:
            attn_zeros = torch.zeros(input.shape).unsqueeze(-1)
        attn_real_and_image = torch.cat([input.squeeze().unsqueeze(-1), attn_zeros], -1)
        # Pytorch version compatible.
        try:
            result = torch.fft(attn_real_and_image, 1)
        except:
            result = torch.fft.fft(input).unsqueeze(-1)
        PSD = torch.norm(result, p=2, dim=1)
        return PSD


class rFFT_MODULE_1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # Pytorch version compatible.
        try:
            result = torch.rfft(input, 1)
        except:
            result = torch.fft.rfft(input).unsqueeze(-1)
        final_result = torch.norm(result, p=2, dim=1)
        return final_result


class Turn_map_into_waves(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn):
        result_list = []
        for i in range(attn.shape[0]):
            result_list.append(self.attn_solo_process(attn[i]).unsqueeze(0))
        result = torch.cat(result_list, 0)
        return result

    def attn_solo_process(self, attn_solo):
        distance_record = [[] for x in range(attn_solo.shape[0])]
        position_check = [[] for x in range(attn_solo.shape[0])]
        for i in range(attn_solo.shape[0]):
            for j in range(i, attn_solo.shape[1]):
                position_check[j - i].append([i, j])
                distance_record[j - i].append(attn_solo[i, j].unsqueeze(0))

        result = []
        for i in range(len(distance_record)):
            result.append(torch.mean(torch.cat(distance_record[i], 0)).unsqueeze(0))
        amassed = torch.cat(result, 0)
        return amassed



