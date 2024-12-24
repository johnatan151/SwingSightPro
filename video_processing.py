import cv2

def open_video(video_path: str):
    """
    Opens the video file and returns the capture object.
    :param video_path: path to the video file
    :return: OpenCV video capture object
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    return cap

def create_video_writer(output_path: str, fps: float, frame_width: int, frame_height: int):
    """
    Creates a VideoWriter object to save the output video.
    :param output_path: path to save the output video
    :param fps: frames per second of the input video
    :param frame_width: width of the video frames
    :param frame_height: height of the video frames
    :return: OpenCV VideoWriter object
    """
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

def process_frame(frame, results):
    """
    Processes the frame by drawing bounding boxes and lines between points.
    :param frame: the input frame
    :param results: the results object containing bounding box info
    :return: the frame with drawn bounding boxes and lines
    """
    frame_with_boxes = results.render()[0]  # Get the frame with bounding boxes drawn
    frame_with_boxes_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

    return frame_with_boxes_bgr

def show_frame(frame, window_name="Frame"):
    """
    Displays the frame in a window.
    :param frame: the frame to display
    :param window_name: name of the display window
    """
    cv2.imshow(window_name, frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False 
    return True
