import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from PIL import Image as Image_PIL
import pyrealsense2 as rs

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import time
import socket
import json
import signal

fx = 917.58190
fy = 915.836303710
ppx = 648.276672363
ppy = 355.7363281

def exit(signum, frame):
    print('You choose to stop me.')
    global stop
    stop = True
    exit()

stop = False

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
def compute_depth(depth_img, bbox):
    x1, y1, x2, y2 = bbox

    O_x1 = max(min(int(x1 + (x2 - x1) / 2.) - 3, IMAGE_WIDTH - 1), 0)
    O_x2 = max(min(int(x1 + (x2 - x1) / 2.) + 3, IMAGE_WIDTH - 1), 0)
    O_y1 = max(min(int(y1 + (y2 - y1) / 2.) - 3, IMAGE_HEIGHT - 1), 0)
    O_y2 = max(min(int(y1 + (y2 - y1) / 2.) + 3, IMAGE_HEIGHT - 1), 0)

    A_x1 = max(min(int(x1 + (x2 - x1) / 4.) - 3, IMAGE_WIDTH - 1), 0)
    A_x2 = max(min(int(x1 + (x2 - x1) / 4.) + 3, IMAGE_WIDTH - 1), 0)
    A_y1 = max(min(int(y1 + (y2 - y1) * 3 / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    A_y2 = max(min(int(y1 + (y2 - y1) * 3 / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    B_x1 = max(min(int(x1 + (x2 - x1) * 3 / 4.) - 3, IMAGE_WIDTH - 1), 0)
    B_x2 = max(min(int(x1 + (x2 - x1) * 3 / 4.) + 3, IMAGE_WIDTH - 1), 0)
    B_y1 = max(min(int(y1 + (y2 - y1) * 3 / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    B_y2 = max(min(int(y1 + (y2 - y1) * 3 / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    C_x1 = max(min(int(x1 + (x2 - x1) / 4.) - 3, IMAGE_WIDTH - 1), 0)
    C_x2 = max(min(int(x1 + (x2 - x1) / 4.) + 3, IMAGE_WIDTH - 1), 0)
    C_y1 = max(min(int(y1 + (y2 - y1) / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    C_y2 = max(min(int(y1 + (y2 - y1) / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    D_x1 = max(min(int(x1 + (x2 - x1) * 3 / 4.) - 3, IMAGE_WIDTH - 1), 0)
    D_x2 = max(min(int(x1 + (x2 - x1) * 3 / 4.) + 3, IMAGE_WIDTH - 1), 0)
    D_y1 = max(min(int(y1 + (y2 - y1) / 4.) - 3, IMAGE_HEIGHT - 1), 0)
    D_y2 = max(min(int(y1 + (y2 - y1) / 4.) + 3, IMAGE_HEIGHT - 1), 0)

    rect_O = depth_img[O_y1:O_y2, O_x1:O_x2]
    dist_O = np.sum(rect_O) / rect_O.size

    rect_A = depth_img[A_y1:A_y2, A_x1:A_x2]
    dist_A = np.sum(rect_A) / rect_A.size

    rect_B = depth_img[B_y1:B_y2, B_x1:B_x2]
    dist_B = np.sum(rect_B) / rect_B.size

    rect_C = depth_img[C_y1:C_y2, C_x1:C_x2]
    dist_C = np.sum(rect_C) / rect_C.size

    rect_D = depth_img[D_y1:D_y2, D_x1:D_x2]
    dist_D = np.sum(rect_D) / rect_D.size

    dist_list = [dist_O, dist_A, dist_B, dist_C, dist_D]
    kp_list = []
    cond_dist_list = []
    for dist in dist_list:
        ko = dist_O / dist
        ka = dist_A / dist
        kb = dist_B / dist
        kc = dist_C / dist
        kd = dist_D / dist
        k_list = [ko, ka, kb, kc, kd]
        kp = 0
        for kx in k_list:
            if 0.9 <= kx <= 1.1:
                kp += 1
        kp_list.append(kp)
        if kp >= 2:
            cond_dist_list.append(dist)
    if len(cond_dist_list) >= 1:
        distance = min(cond_dist_list)
    else:
        distance = dist_O

    # kp_max = max(kp_list)
    # if kp_max >= 3:
    #     distance = dist_list[kp_list.index(kp_max)]
    # else:
    #     distance = dist_O

    return distance

def detect(save_img=False):


    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()


    t0 = time.time()

    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_server_socket.bind(('127.0.0.1', 8084))
    tcp_server_socket.listen(5)
    print('服务端正在等待客户端的连接：')
    signal.signal(signal.SIGINT, exit)
    signal.signal(signal.SIGTERM, exit)

    while (not stop):
        try:
            print('accept\n')
            sock, addr = tcp_server_socket.accept()  # 等待链接,多个链接的时候就会出现问题,其实返回了两个值
            print("socl, addr:", sock, addr)

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
            # Start streaming
            pipeline.start(config)

            sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
            sensor.set_option(rs.option.exposure, 156.000)
            # 深度图像向彩色对齐
            align_to_color = rs.align(rs.stream.color)

            while (not stop):
                start = time.time()
                print('1111111111')
                aa = time.time()
                # try:
                # Wait for a coherent pair of frames: depth and color
                # sock.settimeout(10)
                data = sock.recv(1024)  # 接收数据
                # sock.send(bytes("2112"))
                print("data:", data)
                if (not data):
                    break

                for i in range(2):
                    frames = pipeline.wait_for_frames()
                    print('frams1')
                    frames = align_to_color.process(frames)
                    # print('frams2', frames)
                    depth_frame = frames.get_depth_frame()
                    print('frams3')
                    # print('depth0', depth_frame[204, 482])
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        print('no_frame')
                        continue
                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    r_color_img_data = color_image
                    r_depth_img_data = depth_image

                    color_image = np.ndarray(
                        shape=(color_frame.get_height(), color_frame.get_width(), 3),
                        dtype=np.dtype('uint8'), buffer=r_color_img_data)

                    depth_image = np.ndarray(
                        shape=(color_frame.get_height(), color_frame.get_width()),
                        dtype=np.dtype('uint16'), buffer=r_depth_img_data)
                    color_image = color_image[..., [2, 1, 0]]

                    cv2.imwrite('inference/imgs/1.jpg', color_image)
                    cv2.imwrite('inference/imgs/d1.png', depth_image)
                    # time.sleep(1)

                # Set Dataloader
                vid_path, vid_writer = None, None
                if webcam:
                    view_img = check_imshow()
                    cudnn.benchmark = True  # set True to speed up constant image size inference
                    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
                else:
                    dataset = LoadImages(source, img_size=imgsz, stride=stride)

                # Get names and colors
                names = model.module.names if hasattr(model, 'module') else model.names
                colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

                # Run inference
                if device.type != 'cpu':
                    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                old_img_w = old_img_h = imgsz
                old_img_b = 1

                for path, img, im0s, vid_cap in dataset:
                    # img = cv2.imread('inference/imgs/1.jpg', -1)
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Warmup
                    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                        old_img_b = img.shape[0]
                        old_img_h = img.shape[2]
                        old_img_w = img.shape[3]
                        for i in range(3):
                            model(img, augment=opt.augment)[0]

                    # Inference
                    t1 = time_synchronized()
                    pred = model(img, augment=opt.augment)[0]
                    t2 = time_synchronized()

                    # Apply NMS
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    t3 = time_synchronized()

                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        if webcam:  # batch_size >= 1
                            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                        else:
                            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # img.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
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
                                    print("label:", names[int(cls)])
                                    # if names[int(cls)] == 'bottle' or names[int(cls)] == 'orange' or names[int(cls)] == 'apple':
                                    bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                    print(c1, c2)
                                    xx1 = (int(xyxy[0]) + int(xyxy[2])) / 2
                                    yy1 = (int(xyxy[1]) + int(xyxy[3])) / 2
                                    # zz1 = depth_image[int(xyxy[1]), int(xyxy[0])] / 1000
                                    zz1 = compute_depth(depth_image, bbox) / 1000
                                    if names[int(cls)] == opt.object_name and zz1 < 1:
                                        camera_coordinate1 = ((xx1 - ppx) * zz1 / fx, (yy1 - ppy) * zz1 / fy, zz1)
                                        camera_coordinate2 = (
                                        (int(xyxy[0]) - ppx) * zz1 / fx, (int(xyxy[1]) - ppy) * zz1 / fy, zz1)
                                        camera_coordinate3 = (
                                        (int(xyxy[2]) - ppx) * zz1 / fx, (int(xyxy[3]) - ppy) * zz1 / fy, zz1)

                                        print("camera_cordinate:", camera_coordinate1)
                                        # print(depth_image)
                                        try:
                                            # json_data = json.loads(data.decode())
                                            # print('recive:', json_data)  # 打印接收到的数据
                                            print("point :", camera_coordinate1[0], camera_coordinate1[1],
                                                  camera_coordinate1[2])
                                            xx = float(camera_coordinate1[0])
                                            yy = float(camera_coordinate1[1])
                                            zz = (float(camera_coordinate1[2]) - 0.05)

                                            qx = 0
                                            qy = 0
                                            qz = 0
                                            qw = 1

                                            json_data = {'ret': False,
                                                         'point': [xx, yy, zz, qx, qy, qz, qw]
                                                         }
                                            print("ok!")
                                            json_data1 = json.dumps(json_data)
                                            print('data:', json_data1)
                                            sock.send(bytes(json_data1.encode('utf-8')))  # 然后再发送数据
                                        except Exception as e:
                                            print("ret_False")
                                            err_json = {'ret': False}
                                            err_data = json.dumps(err_json)
                                            sock.send(bytes(err_data.encode('utf-8')))  # 然后再发送数据

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

                                    print("xywh:", xywh)
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                                    print("saving!!!!")
                                    cv2.imwrite('img_yolov7.png', im0)

                        # Print time (inference + NMS)
                        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                        # Stream results
                        if view_img:
                            cv2.imshow(str(p), im0)
                            k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
                            if k == 27:  # 键盘上Esc键的键值
                                cv2.destroyAllWindows()

                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                # cv2.imwrite(save_path, im0)
                                cv2.imwrite('/home/robint01/img_yolov7.png', im0)
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
                                vid_writer.write(im0)

                if save_txt or save_img:
                    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                    #print(f"Results saved to {save_dir}{s}")

                print(f'Done. ({time.time() - t0:.3f}s)')

        except Exception as e:
            # e.what()
            print(e)
            print('error,', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--object-name', type=str, help='source')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
