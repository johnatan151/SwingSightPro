from video_processing import open_video, create_video_writer, show_frame
from model import load_model
from functions import calculate_center, track_line_points, draw_lines
import cv2

# Initialize model and tracker
model = load_model("models/yolov5_best.pt")

# Open video
video_path = "Data/LonestarChip.mov"
cap = open_video(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = create_video_writer('runs/detect/predict/Lonestarchipwconf.mp4', fps, frame_width, frame_height)

# Store flight path
flight = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame and run inference
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=640)
    bbox = results.xyxy[0]
    # Process bounding boxes and track the ball
    coord = []
    for box in bbox:
        x_min, y_min, x_max, y_max = box[:4]
        class_label = int(box[5].item())  
        confidence = box[4].item()

        if class_label == 0 and confidence > 0.40:  
            cx, cy = calculate_center(x_min, y_min, x_max, y_max)
            coord.append((cx, cy))

    # Track the ball's flight path
    for cx, cy in coord:
        flight = track_line_points(flight, cx, cy)

    # Draw the lines and process frame
    processed_frame = draw_lines(frame, flight)

    # Save and display the frame
    out.write(processed_frame)
    if not show_frame(processed_frame):
        break  # Exit loop if 'q' is pressed

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
