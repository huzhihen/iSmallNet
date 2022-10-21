#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import argparse

def Count(pred, mask):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if pred[i][j] != 0 and mask[i][j] != 0:    # TP: 本该是1被分为1 （True positive，TP，真阳）
                count1 += 1
            elif pred[i][j] != 0 and mask[i][j] == 0:  # FP: 本该是0被分为1 （False positive，FP，假阳）
                count2 += 1
            elif pred[i][j] == 0 and mask[i][j] != 0:  # FN: 本该是1被分为0 （False negative，FN，假阴）
                count3 += 1
            elif pred[i][j] == 0 and mask[i][j] == 0:  # TN: 本该是0被分为0 （True negative，TN，真阴）
                count4 += 1
    return count1, count2, count3, count4


def IOU(TP, FP, FN):
    if TP + FP + FN == 0:
        return 0
    return TP / (TP + FP + FN)


def Precision(TP, FP):
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)


def Recall(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def F1measure(Pricision, Recall):
    return 2 * (Pricision * Recall) / (Pricision + Recall)


# Miss Detection
def Md(TP, FP, FN, TN):
    return FN / (TP + FP + FN + TN)


# False Alarm
def Fa(TP, FP, FN, TN):
    return FP / (TP + FP + FN + TN)


def main(args):
    if args.dataset == 'NUAA-SIRST':
        size = 214
    elif args.dataset == 'NUDT-SIRST':
        size = 664
    predpath = "./maps/DNF/" + args.dataset + '/'
    maskpath = "../data/" + args.dataset + "/mask/"
    print(predpath)
    print(maskpath)
    iou_sum = 0
    pre_sum = 0
    rec_sum = 0
    f1_sum  = 0
    md_sum = 0
    fa_sum  = 0

    for name in os.listdir(predpath):
        if args.dataset == 'NUAA-SIRST':
            name1 = name[:-4] + "_pixels0.png"
        elif args.dataset == 'NUDT-SIRST':
            name1 = name
        print(name)
        pred = cv2.imread(predpath + name, 0)
        mask = cv2.imread(maskpath + name1, 0)
        count1, count2, count3, count4 = Count(pred, mask)
        print("count1=", count1)  # TP
        print("count2=", count2)  # FP
        print("count3=", count3)  # FN
        print("count4=", count4)  # TN
        iou = IOU(count1, count2, count3)
        pre = Precision(count1, count2)
        rec = Recall(count1, count3)
        md = Md(count1, count2, count3, count4)
        fa = Fa(count1, count2, count3, count4)

        print("iou=", iou)
        print("pre=", pre)
        print("rec=", rec)
        print("md =", md)
        print("fa =", fa)
        iou_sum += iou
        pre_sum += pre
        rec_sum += rec
        md_sum += md
        fa_sum += fa
    iou_sum /= size
    pre_sum /= size
    rec_sum /= size
    md_sum /= size
    fa_sum /= size
    f1_sum = F1measure(pre_sum, rec_sum)
    print("----------result---------------")
    print("iou_sum=", iou_sum)
    print("pre_sum=", pre_sum)
    print("rec_sum=", rec_sum)
    print("f1_sum =", f1_sum)
    print("md_sum =", md_sum)
    print("fa_sum =", fa_sum)


def parse_args():
    parser = argparse.ArgumentParser(description='DNFNet')
    # dataset
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name:  NUAA-SIRST, NUDT-SIRST')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    main(args)
