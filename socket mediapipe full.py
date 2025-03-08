import mediapipe as mp
import cv2
import socket
import json

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Set up socket connection to C#
HOST = '127.0.0.1'
PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

cap = cv2.VideoCapture(0)

def get_pose_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    data = {"landmarks": {}, "gesture": None}

    if results_pose.pose_landmarks:
        # Extract only important landmarks
        important_landmarks = {11: "Left Shoulder", 12: "Right Shoulder", 
                               23: "Left Hip", 24: "Right Hip", 
                               15: "Left Wrist", 16: "Right Wrist"}

        for i, part in important_landmarks.items():
            landmark = results_pose.pose_landmarks.landmark[i]
            data["landmarks"][part] = {"x": landmark.x, "y": landmark.y}

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

    return data

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    pose_data = get_pose_landmarks(frame)
    sock.sendall(json.dumps(pose_data).encode())

    cv2.putText(frame, pose_data["gesture"] if pose_data["gesture"] else "No Gesture", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Tracking & Gestures', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sock.close()
