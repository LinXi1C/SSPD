# -*- coding:utf-8 -*-
# @File   : test.py
# @Time   : 2022/10/2 20:31
# @Author : Zhang Xinyu
import time
import warnings
import argparse
import torch
import numpy as np
from models.SSPD import SSPD
from utils.hr_calc import hr_cal
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('SSPD testing', add_help=False)
    # Main params.
    parser.add_argument('--frame-path', default='./datasets/p38.npy', type=str,
                        help="""Please specify path to the 'frame_list' as input.""")
    parser.add_argument('--GPU-id', default=0, type=int, help="""Index of GPUs.""")
    parser.add_argument('--pretrained', default='./pretrained/ubfc.pth', type=str, help='pretrained weights path.')
    parser.add_argument('--visual-enable', default='True', type=str, help='visualization or not.')
    return parser


def test(args):
    # ============ Setup logging ... ============
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Start testing at', start_time, end='\n\n')

    # ============ preparing data ... ============
    frame_list = np.load(args.frame_path)[:, :, :, :]
    frame_list_tf = np.zeros([frame_list.shape[0], 128, 128, 3], dtype=np.uint8)
    transform = A.Compose([
        A.Resize(height=128, width=128, interpolation=Image.BICUBIC),
    ])
    for i in range(frame_list.shape[0]):
        frame_list_tf[i] = transform(image=frame_list[i])['image']
    residual_list = np.zeros(
        [frame_list_tf.shape[0]-1, frame_list_tf.shape[1], frame_list_tf.shape[2], frame_list_tf.shape[3]], dtype=np.float32)
    for i in range(residual_list.shape[0]):
        residual_list[i, :, :, :] = frame_list_tf[i+1, :, :, :].astype(np.float32) - frame_list_tf[i, :, :, :].astype(np.float32)
    residual_list = torch.from_numpy(residual_list).permute(3, 0, 1, 2).unsqueeze(0).to(args.GPU_id)

    # ============ building model ... ============
    model = SSPD().to(torch.device(args.GPU_id))
    msg = model.load_state_dict(torch.load(args.pretrained, map_location='cpu'), strict=False)
    print('Pretrained weights found at {}'.format(args.pretrained))
    print('load_state_dict msg: {}'.format(msg), end='\n\n')

    model.eval()

    # ============ testing ... ============
    with torch.no_grad():
        residual_list = residual_list.to(torch.device(args.GPU_id))
        output_rPPG = model(residual_list)[0].cpu().numpy()
        hr_train, rPPG_filtered = hr_cal(output_rPPG, 30)
        print(f'Estimated HR: {hr_train:.4f} bpm')

        if args.visual_enable == 'True':
            font = {'family': 'Arial',
                    'weight': 'bold',
                    'size': 18,
                    }
            plt.rc('font', **font)
            plt.figure(figsize=(12, 5))
            plt.plot(np.arange(len(output_rPPG)), rPPG_filtered, color='red', linewidth=2)
            plt.title(f'Estimated HR: {hr_train}', fontdict=font)
            plt.xlabel('Frame', font)
            plt.ylabel('Amplitude', font)
            plt.tight_layout()
            plt.grid(axis="x", linestyle='--', linewidth=1.5)
            plt.show()

    finish_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Finish testing at', finish_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSPD testing', parents=[get_args_parser()])
    args = parser.parse_args()
    test(args)