import torch
import cv2
from ultralytics import YOLO

print("Torch:", torch.__version__)
print("OpenCV:", cv2.__version__)

model = YOLO("yolov8n.pt")
print("YOLO loaded successfully ✅")
