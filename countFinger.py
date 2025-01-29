import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

# เรียกใช้โมดูล Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# ฟังก์ชันตรวจจับนิ้วที่ยกขึ้น
def count_fingers(hand_landmarks, handedness):
    fingers_up = 0
    finger_tips = [4, 8, 12, 16, 20]  # ตำแหน่งปลายนิ้วใน Mediapipe
    finger_dips = [3, 6, 10, 14, 18]  # ตำแหน่งข้อต่อที่อยู่ก่อนปลายนิ้ว

    # ตรวจสอบว่าหัวแม่มืออยู่ด้านไหน (มือซ้ายหรือขวา)
    if handedness == "Right":
        thumb_up = hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_dips[0]].x
    else:
        thumb_up = hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_dips[0]].x

    if thumb_up:
        fingers_up += 1

    # ตรวจสอบนิ้วที่เหลือ (ถ้าปลายนิ้วอยู่สูงกว่าข้อต่อ แสดงว่ายกขึ้น)
    for i in range(1, 5):
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_dips[i]].y:
            fingers_up += 1

    return fingers_up

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break  # หยุดลูปหากอ่านภาพไม่สำเร็จ

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    left_fingers = 0
    right_fingers = 0

    if results.multi_hand_landmarks:
        for i, handLms in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].label  # ระบุว่ามือไหน (Left / Right)

            fingers = count_fingers(handLms, handedness)

            if handedness == "Left":
                left_fingers = fingers
            else:
                right_fingers = fingers

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # แสดงจำนวนนิ้วที่ยกขึ้น
    cv2.putText(img, f'Left Hand: {left_fingers}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Right Hand: {right_fingers}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", img)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
