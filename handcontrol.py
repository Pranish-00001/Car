import cv2
import mediapipe as mp
import requests
import threading

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

car_ip_address = "localhost"  # Replace with your car's IP address
car_url = f"http://{car_ip_address}/"

cap = cv2.VideoCapture(0)

width, height = 320, 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def send_command(command):
    try:
        requests.get(car_url, params={"State": command})
        print(command)
    except requests.exceptions.RequestException as e:
        print(f"Error sending command to the car: {e}")

def process_video():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Count open fingers
                finger_count = -1
                for finger_tip_id in [4, 8, 12, 16, 20]:
                    tip_landmark = hand_landmarks.landmark[finger_tip_id]
                    if tip_landmark.y < hand_landmarks.landmark[finger_tip_id - 1].y:
                        finger_count += 1

                # Draw landmarks on the hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Send commands based on finger count
                if finger_count == 0:
                    send_command("F")  # Go ahead
                elif finger_count == 1:
                    send_command("B")  # Go back
                elif finger_count == 2:
                    send_command("L")  # Turn left
                elif finger_count == 3:
                    send_command("R")  # Turn right
                elif finger_count == 4:
                    send_command("S")  # Stop

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Create a separate thread for processing video
video_thread = threading.Thread(target=process_video)
video_thread.start()
