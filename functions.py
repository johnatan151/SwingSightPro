# utils.py
import numpy as np
import cv2


def calculate_center(x_min, y_min, x_max, y_max):
    """
    Calculate the center of the bounding box.
    :param x_min, y_min, x_max, y_max: Coordinates of the bounding box
    :return: tuple (cx, cy) representing the center of the box
    """
    cx = int((x_min + x_max) / 2)
    cy = int((y_min + y_max) / 2)
    return cx, cy

def track_line_points(line_points, cx, cy):
    """
    Adds the calculated center of the bounding box to the list of line points.
    :param line_points: List of points (previously tracked points)
    :param cx, cy: Current coordinates to be added
    :return: Updated list of line points
    """
    line_points.append((cx, cy))
    return line_points

def draw_lines(frame, line_points):
    """
    Draws lines between successive points in the line_points list.
    :param frame: the frame on which lines will be drawn
    :param line_points: list of points to connect with lines
    :return: frame with lines drawn
    """
    for i in range(1, len(line_points)):
        cv2.line(frame, line_points[i - 1], line_points[i], color=(255, 0, 0), thickness=10)
    return frame
