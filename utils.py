#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import argparse
import numpy as np

def split_map(args):
    datapath = './data/' + args.dataset
    print(datapath)
    for name in os.listdir(datapath+'/mask'):
        mask = cv2.imread(datapath+'/mask/'+name,0)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        if not os.path.exists(datapath+'/body-origin/'):
            os.makedirs(datapath+'/body-origin/')
        cv2.imwrite(datapath+'/body-origin/'+name, body)

        if not os.path.exists(datapath+'/detail-origin/'):
            os.makedirs(datapath+'/detail-origin/')
        cv2.imwrite(datapath+'/detail-origin/'+name, mask-body)


def parse_args():
    parser = argparse.ArgumentParser(description='DNFNet')
    # dataset
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name:  NUAA-SIRST, NUDT-SIRST')
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    split_map(args)
