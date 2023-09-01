import argparse
import time
from pathlib import Path
import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from PIL import Image as Image_PIL

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

fx = 917.58190
fy = 915.836303710
ppx = 648.276672363
ppy = 355.7363281
def detect(save_img=True):
    ttt = time.time()
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = True 
    #not opt.nosave and not source.endswith('.txt')  # save inference images �ж��Ƿ��ͼƬ���б���?
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))#�ж������ǲ�������ͷ���ж�����ΪһЩ����������

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run <opt.rpject>:����·��;<opt.name>:"expn"�ļ�������<increment_path>:�ı��ļ���ĩβ�����?;<exist_ok>:��·���������Ƿ񴴽�
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()#������־
    device = select_device(opt.device)#ѡ�����ã�Ĭ��Ϊcpu
    half = device.type != 'cpu'  # half precision only supported on CUDA     ���ø��������ȣ�ͨ���豸�жϣ�
 
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model ģ�ͼ���
    stride = int(model.stride.max())  # model stride    �²������?
    imgsz = check_img_size(imgsz, s=stride)  # check img_size    �ж�����ͼƬ�ĳߴ��Ƿ���32�ı���

    if trace:
        model = TracedModel(model, device, opt.img_size)#ת��ģ�ͣ����� traced_model.pt�ļ�
    if half:
        model.half()  # to FP16 ���������ȣ��뾫�ȣ�

    # Second-stage classifier
    classify = False
    if classify:#������Ŀ�����ٷ��࣬��߷��ྫ��?
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    print(f'init Done. ({time.time() - ttt:.3f}s)')
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)#���ݼ���

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names#�ó���������
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]#Ϊÿ�����ֲ����������������ɫ��?

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Ԥ��
    old_img_w = old_img_h = imgsz
    old_img_b = 1




    for path, img, im0s, vid_cap in dataset:#pathͼƬ·����img���ų�640���ͼƬ��im0sԭͼ��vid_cap�Ƿ�����Ƶ�ı�־
        t0 = time.time()
        img = torch.from_numpy(img).to(device)#�ı�ͼƬ��ʽ
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:#�жϼ�ά
            img = img.unsqueeze(0)#��һ��ά��

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]#��ͼƬ����model�н�������
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detection
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path ����·��
            save_path = str(save_dir / p.name)  # img.jpg  ���·��?
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()#��img�ϵ����껹ԭ��im0

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):#�����ս������ͼ��?
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        # color_image = np.array(Image_PIL.open('doc/example_data/color_start.png'),
                        #                        dtype=np.float32) / 255.0
                        depth_image = np.array(Image_PIL.open('inference/imgs/d1.png'))
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        label = f'{names[int(cls)]} {conf:.2f}'

                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        # print(c1, c2)
                        xx1 = (int(xyxy[0]) + int(xyxy[2])) / 2
                        yy1 = (int(xyxy[1]) + int(xyxy[3])) / 2
                        zz1 = depth_image[int(xyxy[1]), int(xyxy[0])] / 1000

                        camera_coordinate1 = ((xx1 - ppx) * zz1 / fx, (yy1 - ppy) * zz1 / fy, zz1)
                        # print("label:", label)
                        # print("camera_cordinate:", camera_coordinate1)
                        # print(depth_image)

                        # vel_msg.x = (x1 + x2) / 2
                        # vel_msg.y = (y1 + y2) / 2
                        # # vel_msg.y = y1
                        # vel_msg.z = dist1[i]
                        # print("original:", vel_msg)
                        # camera_coordinate = ((vel_msg.x - ppx) * vel_msg.z / fx, (vel_msg.y - ppy) * vel_msg.z / fy, vel_msg.z)
                        # # camera_coordinate = rs.rs2_deproject_pixel_to_point(color_intrin_part, [vel_msg.x, vel_msg.y], vel_msg.z)
                        # vel_msg.x = camera_coordinate[0] / 1000
                        # vel_msg.y = camera_coordinate[1] / 1000
                        # vel_msg.z = camera_coordinate[2] / 1000
                        # print("real_coordinate:", vel_msg)

                        # print("xywh:", xywh)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                # print('view')
                cv2.imshow(str(p), im0)
                print(f'Done. ({time.time() - t0:.3f}s)')
                k = cv2.waitKey(1)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms�?? 0代表一直等�??
                if k == 27:  # 键盘上Esc键的键�?
                    cv2.destroyAllWindows()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    # cv2.imwrite(save_path, im0)  #���浽���·��?
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)#����ÿһ֡ͼƬ

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()#��������������
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')#���Ŷȱ߽�����
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')#�����ȱ߽�����
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')#����������
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')#����Ҫ����������
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')#����Ǽ���ֵ���Ʒ���?
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()#�Ѳ�������opt
    # print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():#�������ݶȣ������������ݶȣ�
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
