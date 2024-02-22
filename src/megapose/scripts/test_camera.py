# import the necessary packages
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import json
# import internal functions
from megapose.config import LOCAL_DATA_DIR
from megapose.scripts.run_inference_on_example import *

def detect_objects(image):
     model = fasterrcnn_resnet50_fpn(pretrained=True)
     model.eval()

     # Convert the image to a tensor
     image_tensor = F.to_tensor(image)
     # Add extra dimension to the image tensor
     image_tensor = image_tensor.unsqueeze(0)
     # Make the prediction
     with torch.no_grad():
          prediction = model(image_tensor)
     
     # Convert the prediction to a Python dictionary
     #prediction_dict = {key: prediction[0][key].tolist() for key in prediction[0]}
     prediction_list = [{"label": str(label.item()), "bbox_modal": bbox.tolist()} for label, bbox, score in zip(prediction[0]["labels"], prediction[0]["boxes"], prediction[0]["scores"]) if score.item() > 0.8]
     return prediction_list
#try:
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
# Configure streams
config = rs.config()
#config.enable_stream()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
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
print("Frames received")
# Convert images to numpy arrays
depth_image = np.asanyarray(depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())
print("rgb: ",color_image.shape)
print("depth: ",depth_image.shape)
# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

# Stack both images horizontally
#images = np.hstack((color_image, depth_colormap))

# Show images
example_dir = LOCAL_DATA_DIR / "examples" / "test_camera"
cv2.imwrite(f"{example_dir}/image_depth.png", depth_image)
cv2.imwrite(f"{example_dir}/image_rgb.png", color_image)
#cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#cv2.imshow('RealSense', color_image)
#input("Press Enter to continue...")
#cv2.waitKey(1)

# Object detection
prediction = detect_objects(color_image)
# Save the prediction to a JSON file
with open(f"{example_dir}/inputs/object_data.json", 'w') as f:
    json.dump(prediction, f)
print(prediction)

make_detections_visualization(example_dir)

# Object Pose Estimation
#model_name = "megapose-1.0-RGB-multi-hypothesis"
#run_inference(example_dir, model_name)


#except:
#     print("An error occured")
