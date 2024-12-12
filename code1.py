import cv2
import numpy as np
import os


# This code is used to detect the counters in the video and save the images and video in the output directory

# Function to stabilize the video
def stabilize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    transforms = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = np.mean(flow, axis=(0, 1))
        transforms.append((dx, dy))
        prev_gray = gray

    cap.release()
    return transforms

# Function to apply stabilization to the video
def apply_stabilization(video_path, transforms):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter('stabilized_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    for dx, dy in transforms:
        ret, frame = cap.read()
        if not ret:
            break
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        stabilized_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
        out.write(stabilized_frame)

    cap.release()
    out.release()

# Function to detect counters
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
                counter_contours.append(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw detected counters

    return frame  # Return the frame with drawn counters

# Main execution for multiple videos
video_paths = [
    '/Users/kunalsingh/Downloads/SampleVideos/More videos/E1.mp4'

]  # Update this list with your actual video paths
for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video file at {video_path}")
        continue

    # Create directories to save images and video for each clip
    output_dir = f'detected_counters_images_{os.path.basename(video_path).split(".")[0]}'
    os.makedirs(output_dir, exist_ok=True)

    # Video writer for output video in MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(f'detected_counters_output_{os.path.basename(video_path).split(".")[0]}.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Capture 1 frame per second

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:  # Save 1 frame per second
            detected_frame = detect_counters(frame)
            image_path = os.path.join(output_dir, f'detected_counter_{frame_count // frame_interval}.png')
            cv2.imwrite(image_path, detected_frame)

        # Write the detected frame to the output video
        detected_frame = detect_counters(frame)
        out_video.write(detected_frame)

        frame_count += 1

    cap.release()
    out_video.release()
    print(f"Images saved in '{output_dir}' directory and video saved as 'detected_counters_output_{os.path.basename(video_path).split('.')[0]}.mp4'.")

print("Processing complete for all videos.")
