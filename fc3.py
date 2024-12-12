import cv2
import numpy as np

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

# Function to process video clips
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    best_box = None

    # Initialize background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()

    for frame_count in range(3):  # Process only the first 3 frames
        ret, frame = cap.read()
        if not ret:
            break

        # Detect counters and get bounding boxes
        counter_contours = detect_counters(frame)

        # For this example, we assume the first bounding box is the best
        if counter_contours:
            best_box = counter_contours[0]  # Select the first box as the best

    # Use the best bounding box for the rest of the video
    if best_box is not None:
        print("Best bounding box:", best_box)

        # Define the region underneath the counter
        region_x = best_box[0]
        region_y = best_box[1] + best_box[3]
        region_w = best_box[2]
        region_h = 100  # Define the height of the region as needed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply background subtraction
            fg_mask = back_sub.apply(frame)

            # Find contours in the foreground mask
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Adjust area threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)

                    # Check if the contour is within the region
                    if (region_x < x < region_x + region_w) and (region_y < y < region_y + region_h):
                        # Highlight the person in a different color
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Draw the best bounding box on the frame
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[0] + best_box[2], best_box[1] + best_box[3]), (0, 255, 0), 2)

          
            cv2.rectangle(frame, (region_x, region_y), (region_x + region_w, region_y + region_h), (255, 0, 0), 2)

            # Display the frame
            cv2.imshow('Counter and Region', frame)

            # Exit on 'ESC' key
            if cv2.waitKey(30) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = '/Users/kunalsingh/Downloads/SampleVideos/dataset/Thc s11.mp4'
    process_video(video_path)
