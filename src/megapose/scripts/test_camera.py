# import the necessary packages
import pyrealsense2 as rs
import numpy as np
import cv2
# import internal functions
from megapose.config import LOCAL_DATA_DIR
from .run_inference_on_example import *


try:
     # Create a context object. This object owns the handles to all connected realsense devices
     pipeline = rs.pipeline()
     # Configure streams
     config = rs.config()
     config.enable_stream()
     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

     # Start streaming
     pipeline.start(config)
    
     while True:
          # Wait for a coherent pair of frames: depth and color
          frames = pipeline.wait_for_frames()
          depth_frame = frames.get_depth_frame()
          color_frame = frames.get_color_frame()
          if not depth_frame or not color_frame:
               continue
          else :
               break
     # Convert images to numpy arrays
     depth_image = np.asanyarray(depth_frame.get_data())
     color_image = np.asanyarray(color_frame.get_data())

     # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
     depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

     # Stack both images horizontally
     images = np.hstack((color_image, depth_colormap))

     # Show images
     example_dir = LOCAL_DATA_DIR / "examples" / "test_camera"
     cv2.imwrite('test_depth.png', depth_image)
     cv2.imwrite('test_rgb.png', color_image)
     cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
     cv2.imshow('RealSense', images)
     cv2.waitKey(1)

     # Object detection
     # TODO: implement object detection
     make_detections_visualization(example_dir)
     # Object Pose Estimation
     model_name = "megapose-1.0-RGB-multi-hypothesis"
     run_inference(example_dir, model_name)


except:
     print("An error occured")
