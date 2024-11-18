import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
cap.set(cv2.CAP_PROP_FPS, 30)

image_label = 3
hand_type = 'l'

# Wait for the camera to warm up
for i in range(100):
    cap.read()

image_count = 0  # Counter for saved images
target_count = 15000 # Target number of images

while cap.isOpened() and image_count < target_count:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    # Initialize black_frame in case no hand is detected
    h, w, _ = frame.shape
    black_frame = np.zeros((h, w), dtype=np.uint8)

    # If hands are detected, create an image with only the hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks as white points on the black image
            landmark_points = []
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmark_points.append((x, y))
                cv2.circle(black_frame, (x, y), 5, (255), -1)  # Draw a filled white circle for each point

            # Draw lines connecting the landmarks
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                cv2.line(black_frame, start_point, end_point, (255), 2)  # Draw white line between points

            # Resize to a specified resolution (e.g., 200x200 pixels)
            resized_landmarks = cv2.resize(black_frame, (200, 200))

            # Save the processed image with a unique filename
            output_path = f"images/{image_label}/pp_{hand_type}hand_landmarks{image_count:05d}.png"
            cv2.imwrite(output_path, resized_landmarks)
            print(f"Saved image {image_count + 1}/{target_count} to {output_path}")
            image_count += 1

            # Exit loop if the target number of images is reached
            if image_count >= target_count:
                break

    # Show the camera output with only the landmarks and connections
    cv2.imshow("Landmarks Output - Adjust Your Hand Position", black_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Clean up MediaPipe resources
hands.close()
