# model.py
import torch
import cv2

def load_model(model_path):
    """
    Loads the custom YOLOv5 model from the given path.
    :param model_path: path to the model file
    :return: YOLOv5 model
    """
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path) 
    model.eval()  # Set to evaluation mode
    return model

def run_inference(model, frame):
    """
    Runs inference on a single frame.
    :param model: the loaded YOLOv5 model
    :param frame: the input frame to process
    :return: results object with detected bounding boxes
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=640)
    return results
