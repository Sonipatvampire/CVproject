import cv2
import numpy as np

def detect_counters(frame):


    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counter_contours = []

    # Filter contours based on area and aspect ratio
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            if 1.0 < aspect_ratio < 5.0:  # Filter for counter-like shapes
                counter_contours.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Combine SIFT keypoints with contour information
    filtered_keypoints = []
    for kp in keypoints:
        x, y = map(int, kp.pt)
        # Only keep keypoints that are within or near detected counter regions
        for (cx, cy, cw, ch) in counter_contours:
            if cx-10 <= x <= cx+cw+10 and cy-10 <= y <= cy+ch+10:
                filtered_keypoints.append(kp)
                break

    return filtered_keypoints

def define_regions(keypoints):
    # Define projective lines and regions based on keypoints
    if len(keypoints) < 4:  # Need at least 4 points to define regions
        return []
    
    # Convert keypoints to numpy array of points
    points = np.float32([kp.pt for kp in keypoints])
    
    # Sort points by x coordinate to get left-to-right ordering
    points = points[points[:, 0].argsort()]
    
    # Split points into left and right halves
    mid = len(points) // 2
    left_points = points[:mid]
    right_points = points[mid:]
    
    # Sort each half by y coordinate to get top-to-bottom ordering
    left_points = left_points[left_points[:, 1].argsort()]
    right_points = right_points[right_points[:, 1].argsort()]
    
    # Define regions using quadrilaterals formed by corresponding points
    regions = []
    for i in range(len(left_points)-1):
        if i < len(right_points)-1:
            # Create quadrilateral region using 4 points
            region = np.float32([
                left_points[i],      # Top left
                right_points[i],     # Top right
                right_points[i+1],   # Bottom right
                left_points[i+1]     # Bottom left
            ])
            regions.append(region)
    
    return regions

def apply_homography(frame, regions):
    # Calculate homography and apply perspective transformation
    if not regions:
        return frame
        
    transformed_frame = frame.copy()
    
    # Define standard rectangle size for transformed regions
    std_width = 200
    std_height = 100
    dst_points = np.float32([[0, 0], [std_width, 0], 
                            [std_width, std_height], [0, std_height]])
    
    # Apply homography to each region
    for region in regions:
        # Calculate homography matrix
        H, _ = cv2.findHomography(region, dst_points)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(frame, H, (std_width, std_height))
        
        # Calculate the centroid for overlay
        centroid_x = int(np.mean(region[:, 0]))
        centroid_y = int(np.mean(region[:, 1]))
        
        # Ensure the centroid is within the bounds of the frame
        centroid_x = max(0, min(centroid_x, frame.shape[1] - 1))
        centroid_y = max(0, min(centroid_y, frame.shape[0] - 1))
        
        # Define the top-left corner of the overlay
        top_left_x = max(0, centroid_x - std_width // 2)
        top_left_y = max(0, centroid_y - std_height // 2)
        
        # Ensure the overlay fits within the frame
        if (top_left_x + std_width <= frame.shape[1] and
            top_left_y + std_height <= frame.shape[0]):
            
            # Create an alpha mask for blending
            alpha = 0.5  # Blending factor
            overlay = transformed_frame[top_left_y:top_left_y + std_height, top_left_x:top_left_x + std_width]
            blended = cv2.addWeighted(warped, alpha, overlay, 1 - alpha, 0)
            
            # Place the blended image back onto the frame
            transformed_frame[top_left_y:top_left_y + std_height, top_left_x:top_left_x + std_width] = blended
    
    return transformed_frame

def count_people(transformed_frame, regions):
    # Detect and count people in each region
    counts = []
    
    # Create HOG descriptor for people detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Process each region
    for region in regions:
        # Extract region from transformed frame using mask
        mask = np.zeros(transformed_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(region)], 255)
        roi = cv2.bitwise_and(transformed_frame, transformed_frame, mask=mask)
        
        # Detect people in the region
        boxes, weights = hog.detectMultiScale(
            roi,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        
        # Count unique detections (filter overlapping boxes)
        if len(boxes) > 0:
            pick = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                weights.tolist(),
                score_threshold=0.3,
                nms_threshold=0.4
            )
            count = len(pick)
        else:
            count = 0
            
        counts.append(count)
    
    return counts

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = detect_counters(frame)
        regions = define_regions(keypoints)
        transformed_frame = apply_homography(frame, regions)
        counts = count_people(transformed_frame, regions)
        
        # Display results
        # Draw regions and counts on the frame
        for i, (region, count) in enumerate(zip(regions, counts)):
            # Draw region polygon
            cv2.polylines(frame, [np.int32(region)], True, (0, 255, 0), 2)
            
            # Calculate centroid of region for text placement
            centroid = (int(np.mean(region[:, 0])), int(np.mean(region[:, 1])))
            
            # Draw count
            cv2.putText(frame, 
                       f'Count: {count}',
                       (centroid[0] - 40, centroid[1]),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8,
                       (0, 0, 255),
                       2)
        
        # Show the frame
        cv2.imshow('People Counter', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Example usage
process_video('/Users/kunalsingh/Downloads/SampleVideos/More videos/E1.mp4')