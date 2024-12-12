import cv2
import numpy as np

# Load the template image (the counter you want to detect)
template = cv2.imread('/Users/kunalsingh/Downloads/SampleVideos/dataset/image.png')
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors for the template
keypoints_template, descriptors_template = sift.detectAndCompute(gray_template, None)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Open the video file
video_path = '/Users/kunalsingh/Downloads/SampleVideos/dataset/Thc f1.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # Calculate the interval for 1 FPS

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process only every nth frame (where n is the frame interval)
    if frame_count % frame_interval == 0:
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors in the current frame using SIFT
        keypoints_frame, descriptors_frame = sift.detectAndCompute(gray_frame, None)

        # Match descriptors
        matches = bf.match(descriptors_template, descriptors_frame)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography if enough matches are found
        if len(matches) >= 4:
            homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

            # Define the corners of the template
            h, w = gray_template.shape
            corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32').reshape(-1, 1, 2)

            # Transform corners to the current frame
            transformed_corners = cv2.perspectiveTransform(corners, homography_matrix)

            # Draw the region under the counter
            cv2.polylines(frame, [np.int32(transformed_corners)], isClosed=True, color=(0, 255, 0), thickness=3)
            cv2.putText(frame, "Counter", (int(transformed_corners[0][0][0]), int(transformed_corners[0][0][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the matched frame
        cv2.imshow('Detected Counters', frame)

        # Exit on 'ESC' key
        if cv2.waitKey(30) & 0xFF == 27:
            break

    frame_count += 1

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()