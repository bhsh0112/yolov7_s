from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from tools.pytorch2caffe.read_model import *
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, default='files/config.yaml')
parser.add_argument('--snapshot', type=str, default='files/model.pth')
parser.add_argument('--video_name', type=str, default='files/IMG_1471.MOV')
args = parser.parse_args()

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('MOV') :
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


if __name__ == '__main__':
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    chekpoint = torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu())
    model.load_state_dict(chekpoint)
    model.eval()
    # pre_dict = remove_fc("/home/omnisky/downloads/robint_share/SiameseRPN/model.pth")
    # build tracker
    tracker = build_tracker(model)

    for name, param in model.named_parameters():

        weight = param.detach().numpy()
        print(name, weight.shape)
    print("**"*80)
    for k,v in chekpoint.items():
        print(k, v.shape)
    save_pytorch_weights(model, "tools/cache/weights/")
    extract_bn_weights(chekpoint)
