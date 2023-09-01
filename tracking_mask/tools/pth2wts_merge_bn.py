from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import time
import cv2
import torch
import numpy as np
from glob import glob
import struct

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

@torch.no_grad()
def fuse_conv_and_bn(conv, bn):
    # 初始化
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    # 融合层的权重初始化(W_bn*w_conv(卷积的权重))
    fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )

    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0) )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # 融合层偏差的设置
    fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )

    return fusedconv


torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', default='/home/omnisky/programfiles/tracking/pysot/files/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default='/home/omnisky/programfiles/tracking/pysot/files/model.pth', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
args = parser.parse_args()

# load config
cfg.merge_from_file(args.config)
cfg.CUDA = torch.cuda.is_available()
# device = torch.device('cuda' if cfg.CUDA else 'cpu')
# create model
model = ModelBuilder()

# load model
model.load_state_dict(torch.load(args.snapshot,
    map_location=lambda storage, loc: storage.cpu()))
# model.eval().to(device)

# build tracker
tracker = build_tracker(model)


##### generate pairs to merge BN ########

conv_bn_pairs = {}
bn_param_count = 0
no_bias_conv_count = {}
for k, v in model.state_dict().items():
    layer_strs = k.split(".")
    layer_name = "model."
    for layer_str in layer_strs[:-3]:
        layer_name += layer_str + "."
    layer_name += layer_strs[-3]
    layer_inner_id = int(layer_strs[-2])

    layer_name_prev = layer_name + "[{}]".format(layer_inner_id-1)
    layer_name += "[{}]".format(layer_inner_id)

    type_name = str(type(eval(layer_name)))
    assert ("BatchNorm" in type_name) or ("Conv" in type_name)
    if "BatchNorm" in type_name:
        assert "Conv" in str(type(eval(layer_name_prev)))
        conv_bn_pairs[layer_name_prev] = [layer_name, False]
        bn_param_count += 1
        conv = eval(layer_name_prev)
        if conv.bias is None:
            no_bias_conv_count[layer_name_prev] = 1

        # print(k, v.cpu().numpy().mean())

##### merge BN and save merged weights ########

wts_len = len(model.state_dict().keys()) - bn_param_count + len(no_bias_conv_count)

wrt_count = 0
with open('tools/tracker.wts', 'w') as f:
    f.write('{}\n'.format(wts_len))
    for k, v in model.state_dict().items():

        layer_strs = k.split(".")
        layer_name = "model."
        for layer_str in layer_strs[:-3]:
            layer_name += layer_str + "."
        layer_name += layer_strs[-3]
        layer_inner_id = int(layer_strs[-2])

        layer_name += "[{}]".format(layer_inner_id)

        type_name = str(type(eval(layer_name)))
        if "BatchNorm" in type_name:
            continue
        assert "Conv" in type_name

        if layer_name in conv_bn_pairs:
            is_merged = conv_bn_pairs[layer_name][1]
            if is_merged:
                continue
            conv = eval(layer_name)
            bn = eval(conv_bn_pairs[layer_name][0])
            fused_conv_layer = fuse_conv_and_bn(conv, bn)
            conv_bn_pairs[layer_name][1] = True
            # if conv.bias is None:
            #     break
            save_fuse_layer_name = k.split(".weight")[0].split(".bias")[0]
            for f_key, f_value in fused_conv_layer.state_dict().items():
                save_key = save_fuse_layer_name + ".{}".format(f_key)
                wrt_count += 1
                vr = f_value.reshape(-1).cpu().numpy()
                f.write('{} {} '.format(save_key, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f',float(vv)).hex())
                f.write('\n')
            print(save_key, fused_conv_layer.padding, fused_conv_layer.groups)

        else:
            # print(save_key, fused_conv_layer.padding)
            wrt_count += 1
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f',float(vv)).hex())
            f.write('\n')
