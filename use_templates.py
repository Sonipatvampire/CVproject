import cv2
import numpy as np


# Function to process video using templates
def process_video_with_templates(video_path, templates):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use SIFT to find keypoints and descriptors in the current frame
        sift = cv2.SIFT_create()
        keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

        # Create a BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        for template in templates:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)

            # Match descriptors
            matches = bf.match(descriptors_template, descriptors_frame)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw matches
            matched_frame = cv2.drawMatches(template, keypoints_template, frame, keypoints_frame, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display the matched frame
            cv2.imshow('Detected Counters', matched_frame)

        # Exit on 'ESC' key
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = '/Users/kunalsingh/Downloads/SampleVideos/dataset/Thc s11.mp4'
    templates = [cv2.imread(f'detected_counter_{i}.png') for i in range(6)]  # Load saved templates
    process_video_with_templates(video_path, templates) 