import mediapipe as mp
import cv2
import json

# Initialize MediaPipe Pose & Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

def get_pose_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    data = {"landmarks": {}, "gesture": None}

    # Extract only key body landmarks
    if results_pose.pose_landmarks:
        important_landmarks = {11: "Left Shoulder", 12: "Right Shoulder", 
                               23: "Left Hip", 24: "Right Hip", 
                               15: "Left Wrist", 16: "Right Wrist"}

        for i, part in important_landmarks.items():
            landmark = results_pose.pose_landmarks.landmark[i]
            data["landmarks"][part] = {"x": round(landmark.x, 3), "y": round(landmark.y, 3)}

    # Detect hand gestures
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Detect Swipe
            if index.x < wrist.x - 0.1:
                data["gesture"] = "SWIPE_LEFT"
            elif index.x > wrist.x + 0.1:
                data["gesture"] = "SWIPE_RIGHT"

            # Detect Thumbs Up/Down
            if thumb.y < wrist.y - 0.1:
                data["gesture"] = "THUMBS_UP"
            elif thumb.y > wrist.y + 0.1:
                data["gesture"] = "THUMBS_DOWN"

    return data, results_pose, results_hands

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get landmarks and gestures
    pose_data, results_pose, results_hands = get_pose_landmarks(frame)
    
    # Print data in the terminal
    print(json.dumps(pose_data, indent=2))

    # Visualization - Draw landmarks on the frame
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display detected gesture on screen
    cv2.putText(frame, pose_data["gesture"] if pose_data["gesture"] else "No Gesture", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show output window
    cv2.imshow('Skeleton & Gesture Tracking', frame)

    # Stop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
