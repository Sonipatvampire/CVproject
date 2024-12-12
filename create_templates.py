import cv2
import numpy as np
import os

# This code is used to detect the counters in the video and save the images and video in the output directory

# Function to detect counters and return bounding boxes
def detect_counters(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter_contours = []

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            if 1.0 < aspect_ratio < 5.0:  # Wider range to capture more counters
                counter_contours.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw detected counters

    return counter_contours

# Function to create templates from the first three frames
def create_templates(video_path):
    cap = cv2.VideoCapture(video_path)
    templates = []
    frame_count = 0

    while cap.isOpened() and frame_count < 3:  # Process only the first 3 frames
        ret, frame = cap.read()
        if not ret:
            break

        # Detect counters and get bounding boxes
        bounding_boxes = detect_counters(frame)

        # Extract templates from bounding boxes
        for i, (x, y, w, h) in enumerate(bounding_boxes):
            template = frame[y:y+h, x:x+w]
            templates.append(template)
            cv2.imwrite(f'detected_counter_{i}.png', template)  # Save template as image

        frame_count += 1

    cap.release()
    return templates

# Example usage
if __name__ == "__main__":
    video_path = '/Users/kunalsingh/Downloads/SampleVideos/dataset/Thc f1.mp4'
    templates = create_templates(video_path)
