import cv2
import numpy as np

# This code is used to detect the counters in the video and save the images and video in the output directory

# Function to detect counters and return bounding boxes
def detect_counters(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Use SIFT to find keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # For demonstration, let's assume we find some bounding boxes (you'll need to implement your detection logic)
    # Here, we create dummy bounding boxes for illustration
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

# Function to process video clips
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    best_template = None
    best_box = None

    for frame_count in range(3):  # Process only the first 3 frames
        ret, frame = cap.read()
        if not ret:
            break

        # Detect counters and get bounding boxes
        counter_contours = detect_counters(frame)

        # Draw bounding boxes and select the best one (for simplicity, we take the first one)
        #for (x, y, w, h) in counter_contours:
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Detected Counters', frame)
        cv2.waitKey(500)  # Show each frame for 500 ms

        # For this example, we assume the first bounding box is the best
        if counter_contours:
            best_box = counter_contours[1]  # Select the first box as the best
            best_template = frame[best_box[1]:best_box[1] + best_box[3], best_box[0]:best_box[0] + best_box[2]]

    # Use the best bounding box for the rest of the video
    if best_box is not None:
        print("Best bounding box:", best_box)

        # Process the rest of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw the best bounding box on the frame
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[0] + best_box[2], best_box[1] + best_box[3]), (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Static Template Detection', frame)

            # Exit on 'ESC' key
            if cv2.waitKey(30) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = '/Users/kunalsingh/Downloads/SampleVideos/More videos/E1.mp4'
    process_video(video_path)


