# -*- coding:utf-8 -*-
# @File   : train.py
# @Time   : 2022/11/28 10:22
# @Author : Zhang Xinyu
import os
import time
import sys
import warnings
import argparse
import math
from pathlib import Path
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from datasets.UBFC import Dataset_UBFC_Offline
from models.SSPD import SSPDL3
from losses.SSPDLoss import SSPDLoss
from utils.utils import cosine_scheduler, LARS, restart_from_checkpoint, get_params_groups
from utils.metric_logger import AverageMeter, ProgressMeter

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('SSPD training', add_help=False)
    # Main params.
    parser.add_argument('--frame-path', default='/home/linxi/data/ubfc-frame-SKD/frame_list', type=str,
                        help="""Please specify path to the 'frame_list' as input.""")
    parser.add_argument('--SSL_length', default=300, type=int, help="""Length of video frames.""")
    parser.add_argument('--output-dir', default='./saved/', type=str, help="""Path to save logs and checkpoints.""")
    parser.add_argument('--GPU-id', default=0, type=int, help="""Index of GPUs.""")
    parser.add_argument('--saveckp-freq', default=1, type=int, help="""Save checkpoint every x epochs.""")
    parser.add_argument('--num-workers', default=8, type=int, help="""Number of data loading workers per GPU.""")
    parser.add_argument('--batch-size', default=12, type=int, help="""batch-size: number of distinct images loaded on GPU.""")
    parser.add_argument('--epochs', default=30, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=1e-3, type=float, help="""Learning rate at the end of linear warmup (highest LR 
                        used during training). The learning rate is linearly scaled with the batch size, and specified here
                        for a reference batch size of 256.""")
    parser.add_argument('--min-lr', default=1e-3, type=float, help="""Target LR at the end of optimization. We use a cosine
                        LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adamw', 'adam', 'sgd', 'lars'], help="""Type of 
                        optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--use_fp16', default='True', type=str, help="""Whether or not to use half precision for training. 
                        Improves training time and memory requirements, but can provoke instability and slight decay of 
                        performance. We recommend disabling mixed precision if the loss is unstable, if reducing the patch 
                        size or if training with bigger ViTs.""")
    parser.add_argument("--warmup-epochs", default=0, type=int, help="""Number of epochs for the linear learning-rate warm
                         up.""")
    parser.add_argument('--weight-decay', default=0, type=float, help="""Initial value of the weight decay. With ViT, a 
                        smaller value at the beginning of training works well.""")
    parser.add_argument('--weight-decay-end', default=0, type=float, help="""Final value of the weight decay. We use a 
                        cosine schedule for WD and using a larger decay by the end of training improves performance for 
                        ViTs.""")
    parser.add_argument('--log-enable', default='True', type=str, help="""Whether or not enable tensorboard and logging.""")
    parser.add_argument('--print-freq', default=1, type=int, help="""Print metrics every x iterations.""")
    parser.add_argument('--log-theme', default='UBFC SSPD-L3', type=str, help="""Annotation for tensorboard.""")
    parser.add_argument('--momentum', default=0.9, type=float, help="""Base EMA parameter for teacher update. 
                        The value is increased to 1 during training with cosine schedule.""")
    parser.add_argument('--checkpoint', default=None, type=str, help="""Path for checkpoint.""")
    return parser


def train(args):
    cudnn.benchmark = True
    # ============ Setup logging ... ============
    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Start training at', start_time, end='\n\n')
    if args.log_enable=='True':
        tb_writer = SummaryWriter(f'./tensorboard/SSPD_GPU{args.GPU_id}',
                                  filename_suffix=f'_SSPD_GPU{args.GPU_id}')
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        logging.basicConfig(filename=f'./logs/SSPD_GPU{args.GPU_id}.log', filemode='w', level=logging.INFO,
                            format='%(levelname)s: %(message)s')
        logging.info('Start training at {}\n'.format(start_time))

        # logging hyperparameters
        logging.info("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args)).items()) + '\n')
    else:
        tb_writer = None
    print("\n".join("%s: %s" % (k, str(v)) for k, v in dict(vars(args)).items()), end='\n\n')

    # ============ preparing data ... ============
    train_set = [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34,
                 35, 36, 37]
    person_name = [rf"p{i}" for i in train_set]
    train_set = Dataset_UBFC_Offline(args.frame_path, person_name, length=args.SSL_length, maxOverlap=0, retain_prob=0.7)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,  # set 'pin_memory=True' if you have enough RAM.
        shuffle=True,
        drop_last=False
    )
    print(f"Data loaded: there are {len(train_set)} videos for training.")

    # ============ building model ... ============
    onlineNet = SSPDL3().to(torch.device(args.GPU_id))
    total = sum([param.nelement() for param in onlineNet.parameters()])
    print("Number of Online parameters: {:.2f}M".format(total / 1e6))
    targetNet = SSPDL3().to(torch.device(args.GPU_id))
    total = sum([param.nelement() for param in targetNet.parameters()])
    print("Number of Target parameters: {:.2f}M".format(total / 1e6), end='\n\n')

    # target.params = online.params
    targetNet.load_state_dict(onlineNet.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients.
    for p in targetNet.parameters():
        p.requires_grad = False  # before optimizer init...

    # ============ preparing loss ... ============
    sspd_loss = SSPDLoss(args)

    # ============ preparing optimizer ... ============
    params_groups = get_params_groups(onlineNet)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, lr=0)  # to use with ViTs
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_groups, lr=0)  # lr is set by scheduler
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = LARS(params_groups)  # to use with convnet and large batches

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16 == 'True':
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )
    momentum_schedule = cosine_scheduler(args.momentum,
                                         1,
                                         args.epochs, len(train_loader),
                                         warmup_epochs=args.warmup_epochs,
                                         start_warmup_value=0.99)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.checkpoint:
        restart_from_checkpoint(
            os.path.join(args.checkpoint),
            run_variables=to_restore,
            online_model=onlineNet,
            target_model=targetNet,
            optimizer=optimizer,
        )
        start_epoch = to_restore["epoch"]
    else:
        start_epoch = to_restore["epoch"]

    print("Starting training !")
    onlineNet.train()
    for epoch in range(start_epoch, args.epochs):
        # ============ training one epoch ... ============
        train_one_epoch(onlineNet, targetNet, sspd_loss, train_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                        epoch, logging, tb_writer, fp16_scaler, args)
        # ============ save weights ... ============
        save_dict = {
            'online_model': onlineNet.state_dict(),
            'target_model': targetNet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args
        }
        if epoch % args.saveckp_freq == 0 or epoch == args.epochs-1:
            torch.save(save_dict, os.path.join(args.output_dir, f'SSPD_SKD_ckp{epoch:04}_GPU{args.GPU_id}.pth'))

    finish_time = time.strftime('%Y-%m-%d %H:%M:%S')
    print('Finish training at', finish_time)


def train_one_epoch(onlineNet, targetNet, sspd_loss, data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, logging, tb_writer, fp16_scaler, args):
    # metric init.
    batch_time_metric = AverageMeter('Time', ':6.3f')
    total_loss_metric = AverageMeter('Total Loss', ':5.3f')
    loss_pyramid_metric = AverageMeter('Loss Pyramid', ':5.3f')
    loss_rPPG_metric = AverageMeter('Loss rPPG', ':5.3f')
    loss_reg_metric = AverageMeter('Loss Reg', ':5.3f')
    loss_freq_metric = AverageMeter('Loss Freq', ':5.3f')
    lr_metric = AverageMeter('lr', ':6.4f')
    wd_metric = AverageMeter('wd', ':6.4f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time_metric,
         total_loss_metric,
         loss_pyramid_metric,
         loss_rPPG_metric,
         loss_reg_metric,
         loss_freq_metric,
         lr_metric,
         wd_metric],
        prefix="Epoch: [{}]".format(epoch),
        log_enable=args.log_enable
    )

    end = time.time()
    for it, (residual_input, residual_input2, path, start_f, end_f, _, overlap) in enumerate(data_loader):
        train_iter = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[train_iter]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[train_iter]

        residual_input = residual_input.to(torch.device(args.GPU_id))
        residual_input2 = residual_input2.to(torch.device(args.GPU_id))

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            online_output = onlineNet(residual_input)
            target_output = targetNet(residual_input2)

            # loss calculate.
            total_loss, loss_pyramid, loss_rPPG, loss_reg, loss_freq, target_SSM_return, online_SSM_return = \
                sspd_loss(target_output, online_output)
            # ================= end of the amp calc. =================

        if args.log_enable=='True':
            with torch.no_grad():
                for b in range(residual_input.shape[0]):  # batch size
                    for p in range(len(online_SSM_return)):
                        tb_writer.add_image(
                            '_'.join(['online', f'epoch{epoch}', path[b].split('_')[0], path[b].split('_')[1], path[b].split('_')[2],
                                      path[b].split('_')[3], str(start_f[b].item()), str(end_f[b].item())]),
                            np.expand_dims((np.array(online_SSM_return[p][b]) * 127.5 + 127.5), axis=0).astype(np.uint8), p)
                    for q in range(len(target_SSM_return)):
                        tb_writer.add_image(
                            '_'.join(['target', f'epoch{epoch}', path[b].split('_')[0], path[b].split('_')[1], path[b].split('_')[2],
                                      path[b].split('_')[3], str(start_f[b].item()), str(end_f[b].item())]),
                            np.expand_dims((np.array(target_SSM_return[q][b]) * 127.5 + 127.5), axis=0).astype(np.uint8), q)
                    fig, ax = plt.subplots(figsize=(20, 5))
                    ax.plot(online_output[1][b].detach().cpu().squeeze().numpy(), label='online rPPG')
                    ax.plot(target_output[1][b].detach().cpu().squeeze().numpy(), label='target rPPG')
                    ax.legend()
                    # plt.show()
                    ax.set_title('_'.join(['rPPG', f'epoch{epoch}', path[b].split('_')[0], path[b].split('_')[1], path[b].split('_')[2],
                                  path[b].split('_')[3], str(start_f[b].item()), str(end_f[b].item())]))
                    tb_writer.add_figure('_'.join(['rPPG', f'epoch{epoch}', f'iter{it}']), fig, b)

        # metric update.
        total_loss_metric.update(total_loss.item(), residual_input.shape[0])
        loss_pyramid_metric.update(loss_pyramid, residual_input.shape[0])
        loss_rPPG_metric.update(loss_rPPG, residual_input.shape[0])
        loss_reg_metric.update(loss_reg, residual_input.shape[0])
        loss_freq_metric.update(loss_freq, residual_input.shape[0])
        lr_metric.update(optimizer.param_groups[0]["lr"], residual_input.shape[0])
        wd_metric.update(optimizer.param_groups[0]["weight_decay"], residual_input.shape[0])

        if not math.isfinite(total_loss.item()):
            print("Total loss is {}, stopping training".format(total_loss.item()))
            progress.display(it)
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            total_loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(total_loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # estimate elapsed time.
        batch_time_metric.update(time.time() - end)
        end = time.time()

        if it % args.print_freq == 0:
            progress.display(it)

        # EMA update for the teacher.
        with torch.no_grad():
            m = momentum_schedule[train_iter]  # momentum parameter
            params_update = {}
            for name, param in onlineNet.named_parameters():
                params_update.update({name:param})
            for name, param in targetNet.named_parameters():
                if name in params_update.keys():
                    param.data.mul_(m).add_((1 - m) * params_update[name].detach().data)

    print('Train Avg: Total Loss: {total_loss.avg:.4f}'.format(total_loss=total_loss_metric))

    # ============ writing logs ... ============
    if args.log_enable=='True':
        logging.info('Train Avg: Total Loss: {total_loss.avg:.4f}'.format(total_loss=total_loss_metric))
        tb_writer.add_scalars('train_loss', {'total_loss': total_loss_metric.avg,
                                             'loss_pyramid': loss_pyramid_metric.avg,
                                             'loss_rPPG': loss_rPPG_metric.avg,
                                             'loss_reg': loss_reg_metric.avg,
                                             'loss_freq': loss_freq_metric.avg,
                                             'learning_rate': lr_metric.avg,
                                             'weight_decay': wd_metric.avg}, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SSPD training', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)