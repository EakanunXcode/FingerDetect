import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break  # หยุดลูปหากอ่านภาพไม่สำเร็จ

    # แปลงเป็นภาพสีเทา และเบลอเพื่อลดสัญญาณรบกวน
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ใช้ Threshold เพื่อแยกส่วนของสีดำ
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    # ค้นหา Contours (เส้นรอบรูป)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    phone_count = 0  # ตัวแปรเก็บจำนวนโทรศัพท์ที่ตรวจพบ

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)

        # กรองเฉพาะวัตถุที่เป็นสี่เหลี่ยมและมีขนาดใหญ่พอ
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0 and w > 50 and h > 100:  # ขนาดคร่าวๆ ของโทรศัพท์
            phone_count += 1
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # วาดกรอบสีเขียวรอบโทรศัพท์

    # แสดงจำนวนโทรศัพท์ที่ตรวจพบ
    cv2.putText(img, f'Phones: {phone_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Phone Detection", img)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
