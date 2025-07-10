import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##########################
wCam, hCam = 640, 480
frameR = 0  # 🔥 FULL tracking area (no margin)
smoothening = 7
##########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Try default webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Hand detector from HandTrackingModule
detector = htm.HandDetector(maxHands=1)

# Get screen size
wScr, hScr = autopy.screen.size()

while True:
    # 1. Read webcam frame
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # 2. Find hands and landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        # 3. Get finger tip positions
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # 4. Check which fingers are up
        fingers = detector.fingersUp()

        # 5. Moving Mode: Only index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            # 6. Convert coordinates to screen space
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # 7. Smoothen movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 8. Move mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 9. Clicking Mode: Both index and middle fingers up
        if fingers[1] == 1 and fingers[2] == 1:
            # 10. Check distance between fingertips
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-5)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Show image
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
