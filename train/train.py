#!/usr/bin/python3
# coding=utf-8

import sys
import datetime
import argparse
import numpy as np

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from tqdm import tqdm
from net import iSmall, Res_CBAM_Block


def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou   = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def miou(pred, mask):
    mini  = 1
    maxi  = 1
    nbins = 1
    predict = (pred > 0).float()
    intersection = predict * ((predict == mask).float())

    area_inter, _ = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred, _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab, _   = np.histogram(mask.cpu(), bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    iou = 1 * area_inter / (np.spacing(1) + area_union)

    return iou.mean()


def train(Dataset, Network, args):
    # dataset
    datapath = '../data/' + args.dataset
    outpath  = './out/' + args.dataset

    # train dataloader
    train_cfg    = Dataset.Config(datapath=datapath, savepath=outpath, mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=1500)
    train_data   = Dataset.Data(train_cfg)
    train_loader = DataLoader(train_data, collate_fn=train_data.collate, batch_size=train_cfg.batch, shuffle=True, pin_memory=True, num_workers=2)

    # test dataloader
    test_cfg    = Dataset.Config(datapath=datapath, mode='test')
    test_data   = Dataset.Data(test_cfg)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    # network
    device = torch.device('cuda:5' if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))
    net = Network(train_cfg, block=Res_CBAM_Block,  nb_filter=[16, 32, 64, 128, 256], block_nums=[2, 2, 2, 2])
    net.to(device)

    # parameter
    base, head = [], []
    for name, param in net.named_parameters():
        head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=train_cfg.lr, momentum=train_cfg.momen, weight_decay=train_cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(train_cfg.savepath)
    global_step    = 0

    # 保存模型的loss参数
    train_loss_all = []
    test_loss_all  = []
    test_miou_all  = []
    iou_max = 0

    for epoch in range(train_cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (train_cfg.epoch + 1) * 2 - 1)) * train_cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (train_cfg.epoch + 1) * 2 - 1)) * train_cfg.lr

        train_loss = 0.0
        train_num  = 0
        train_tbar = tqdm(train_loader)
        net.train()

        for step, (image, mask, body, detail) in enumerate(train_tbar):
            image, mask, body, detail = image.to(device), mask.to(device), body.to(device), detail.to(device)
            outb1, outd1, out1 = net(image)

            lossb1 = F.binary_cross_entropy_with_logits(outb1, body) + iou_loss(outb1, body)
            lossd1 = F.binary_cross_entropy_with_logits(outd1, detail) + iou_loss(outd1, detail)
            loss1  = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)
            loss   = (lossb1 + lossd1 + loss1)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            # loss
            train_loss += loss.item() * image.size(0)
            train_num  += image.size(0)
            train_all = train_loss / train_num

            # log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'lossb1': lossb1.item(), 'lossd1': lossd1.item(), 'loss1': loss1.item()},
                           global_step=global_step)
            train_tbar.set_description('Time:%s | Epoch:%d/%d | train loss=%.6f'
                                       % (datetime.datetime.now(), epoch + 1, train_cfg.epoch, train_all))

        test_loss = 0.0
        test_num  = 0
        test_iou  = 0.0
        test_iou_num = 0
        test_tbar = tqdm(test_loader)
        net.eval()

        with torch.no_grad():
            for step, (image, mask, (H, W), name) in enumerate(test_tbar):
                image, mask, shape = image.to(device), mask.to(device), (H,W)
                outb1, outd1, out1 = net(image, shape)
                loss = iou_loss(out1, mask)
                test_loss += loss.item() * image.size(0)
                test_num  += image.size(0)
                test_all = test_loss / test_num

                out1 = torch.round(torch.sigmoid(out1) * 255)
                mask = mask.unsqueeze(1)
                iou  = miou(out1, mask)
                test_iou     += iou.item() * image.size(0)
                test_iou_num += image.size(0)
                test_iou_all = test_iou / test_iou_num
                test_tbar.set_description('Time:%s | Epoch:%d/%d | test loss=%.6f | test iou=%.6f'
                                           % (datetime.datetime.now(), epoch + 1, train_cfg.epoch, test_all, test_iou_all))

        if test_iou_all > iou_max:
            iou_max = test_iou_all
            torch.save(net.state_dict(), train_cfg.savepath + '/model-best')

        # 计算一个epoch在训练集和验证集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        test_loss_all.append(test_loss / test_num)
        test_miou_all.append(test_iou / test_iou_num)
        print('Epoch:{} | Train Loss:{:.4f} | Test Loss:{:.4f} | Test miou:{:.4f} | iou_max:{:.4f}'.format(epoch + 1, train_loss_all[-1], test_loss_all[-1], test_miou_all[-1], iou_max))
        resultpath = './out/' + args.dataset + '/result.txt'
        with open(resultpath, 'a') as f:
            txt = 'Epoch:{} | Train Loss:{:.4f} | Test Loss:{:.4f} | Test miou:{:.4f} | iou_max:{:.4f}'.format(epoch + 1, train_loss_all[-1], test_loss_all[-1], test_miou_all[-1], iou_max)
            f.write(txt + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='iSmallNet')
    # dataset
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name:  NUAA-SIRST, NUDT-SIRST')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train(dataset, iSmall, args)
