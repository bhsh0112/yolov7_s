import pyrealsense2 as rs
import numpy as np
import cv2
import time


def show_colorizer_depth_img():
  colorizer = rs.colorizer()
  hole_filling = rs.hole_filling_filter()
  filled_depth = hole_filling.process(depth_frame)
  colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
  cv2.imshow('filled depth', colorized_depth)


if __name__ == "__main__":
  # Configure depth and color streams
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgba8, 30)
  # Start streaming
  pipeline.start(config)
  # 深度图像向彩色对齐
  align_to_color = rs.align(rs.stream.color)
  imgname = 0
  count = 0

  try:
    while True:
      # Wait for a coherent pair of frames: depth and color
      frames = pipeline.wait_for_frames()

      frames = align_to_color.process(frames)

      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()
      if not depth_frame or not color_frame:
        continue
      # Convert images to numpy arrays
      depth_image = np.asanyarray(depth_frame.get_data())
      color_image = np.asanyarray(color_frame.get_data())

      r_color_img_data = color_image
      r_depth_img_data = depth_image

      # color_image = np.ndarray(
      #   shape=(color_frame.get_height(), color_frame.get_width(), 3),
      #   dtype=np.dtype('uint8'), buffer=r_color_img_data)
      #
      # depth_image = np.ndarray(
      #   shape=(color_frame.get_height(), color_frame.get_width()),
      #   dtype=np.dtype('uint16'), buffer=r_depth_img_data)
      color_image = color_image[..., [2, 1, 0]]

      print("saved sucessful%d", count)
      fname = str(imgname) + '.jpg'

      cv2.imwrite('/home/robint01/yolov7/inference/imgs/' + fname, color_image,
                  [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 路径换成自己的路径
      dname = str(imgname) + '.png'
      cv2.imwrite('/home/robint01/yolov7/inference/imgs/d' + dname, depth_image)
      # time.sleep(3)


      count = count + 1
      if count >= 1:
        print("change pose", fname)
        time.sleep(2)
        count = 0
      imgname = imgname + 1

      # show_colorizer_depth_img()
      cv2.imshow('filled depth', color_image)
      # Press esc or 'q' to close the image window
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
  finally:
    # Stop streaming
    pipeline.stop()
