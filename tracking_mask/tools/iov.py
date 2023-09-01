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

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', default='/home/omnisky/programfiles/tracking/pysot/files/config.yaml', type=str, help='config file')
parser.add_argument('--snapshot', default="/home/omnisky/programfiles/tracking/pysot/files/model.pth", type=str, help='model name')
# parser.add_argument('--video_names', default='', type=str, help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    print("Extracting frames [{}] ...".format(video_name))
    frames = []
    canvases = []
    cap = cv2.VideoCapture(video_name)
    while (cap.isOpened()):
        success, frame = cap.read()
        if success != True:
            break
        h, w, c = frame.shape
        canvas = np.zeros([h, w*2+100, c])
        canvas[:, :w, :] = frame.copy()
        canvas[:, w:w*2, :] = frame.copy()
        canvas[:, w*2:, :] = 255

        frames.append(frame.copy())
        canvases.append(canvas.copy())   
    return frames, canvases

if __name__ == '__main__':
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    # print(cfg)
    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    im_color = cv2.applyColorMap(np.arange(256).astype('uint8') , cv2.COLORMAP_HSV)[:,0,:]

    trk_root = 

    root = "/raid/Datasets/Challenges/IOV/"
    data_root = root + "test_videos/test/video/"
    raw_cross_root = root + "paper_anno_raw/result/"
    save_root = root + "paper_anno_raw/pysot_track/"
    video_names = os.listdir(data_root)
    for v_id, video_name in enumerate(video_names):
        print("Processed {} %".format(v_id*100/len(video_names)))
        save_dir = save_root + video_name.split(".")[0]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        video_path = data_root + video_name
        frames, canvases = [], []
        frames, canvases = get_frames(video_path)
        cross_file_path = raw_cross_root + video_name.replace(".mp4", ".txt")
        with open(cross_file_path, "r") as cross_f:
            raw_lines = cross_f.readlines()
        cross_events = [l.strip().split() for l in raw_lines]
        color_stride = im_color.shape[0]//len(cross_events) if len(cross_events) !=0 else 0

        starts = {}
        ends = {}
        s_bboxes = {}
        e_bboxes = {}
        trackers = []
        inited = []
        end = []
        colors = []
        for e_id, cross_event in enumerate(cross_events):
            cross_event = [int(c) for c in cross_event]
            trackers.append(build_tracker(model))
            starts[e_id] = cross_event[0]
            bbox = cross_event[1:5]
            s_bboxes[e_id] = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            ends[e_id] = cross_event[6]
            bbox = cross_event[7:11]
            e_bboxes[e_id] = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            inited.append(False)
            end.append(False)
            color = im_color[color_stride*e_id].tolist()
            colors.append(color)
        print("Processing Tracking")
        for f_id, frame in enumerate(frames):

            h, w, c = frame.shape
            c_x = w*2+5
            c_y = 40

            for e_id, s_f_id in starts.items():
                e_f_id = ends[e_id]

                color = colors[e_id]
                cv2.line(canvases[f_id], (c_x, c_y), (c_x+45, c_y), color=color, thickness=5)
                cv2.putText(canvases[f_id], str(e_id+1), (c_x+55, c_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvases[f_id], str(s_f_id), (c_x+5, c_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                cv2.putText(canvases[f_id], str(e_f_id), (c_x+40, c_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                c_y += 40

                if f_id == s_f_id:
                    s_bbox = s_bboxes[e_id]
                    trackers[e_id].init(frame, s_bbox)
                    inited[e_id] = True
                if f_id == e_f_id:
                    s_bbox = s_bboxes[e_id]
                    end[e_id] = True

                if inited[e_id] and not end[e_id]:
                    outputs = trackers[e_id].track(frame)
                    bbox = list(map(int, outputs['bbox']))
                    cv2.rectangle(canvases[f_id], (bbox[0] + w, bbox[1]),
                            (bbox[0]+bbox[2]+w, bbox[1]+bbox[3]), color, 3)

            vis_path = save_dir + "/{}.jpg".format(f_id)
            cv2.imwrite(vis_path, canvases[f_id])
            trk_f.write("{} {} {} {} {}\n".format(n, bbox[0],bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
        trk_f.close()
        # break