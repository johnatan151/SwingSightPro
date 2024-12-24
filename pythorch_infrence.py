import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="models/yolov5_best.pt")  # Local model

# Open video
video_path = "Data/IMG_4106.mov"
cap = cv2.VideoCapture(video_path)

# Get video details
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer to save output video
out = cv2.VideoWriter('runs/detect/predict/Data/karchtracer.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# List to store middle points across frames
line_points = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run inference on the current frame
    results = model(frame_rgb, size=640)
    bbox = results.xyxy[0]
    # Process each detection
    for i, box in enumerate(bbox):  # Access the bounding boxes
       if i % 4 == 0:  # Process every 4th box
        x_min, y_min, x_max, y_max = box[:4]  # Bounding box coordinates
        confidence = box[4].item()  # Confidence score
        class_label = int(box[5].item())  # Class label index

        # Calculate the middle point of the bounding box
        cx = int((x_min + x_max) / 2)
        cy = int((y_min + y_max) / 2)
        # Store the middle point
        line_points.append((cx, cy))

    # Render results on the frame
    frame_with_boxes = results.render()[0]  # Get the frame with bounding boxes drawn
    frame_with_boxes_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

    # Draw lines between successive points
    for i in range(1, len(line_points)):
        cv2.line(frame_with_boxes_bgr, line_points[i - 1], line_points[i], color=(255, 0, 0), thickness=2)

    # Write the frame to the output video
    out.write(frame_with_boxes_bgr)

    # Show the frame (optional)
    cv2.imshow('Frame', frame_with_boxes_bgr)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
