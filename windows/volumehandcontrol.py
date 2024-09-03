import cv2
import time
import math
import numpy as np
import handtrackmodule as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Webcam resolution
wCam, hCam = 640, 480

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector()

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume.iid, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get volume range
volrange = volume.GetVolumeRange()
minvol = volrange[0]
maxvol = volrange[1]
vol = 0
volBar = 400
volPer = 0

ptime = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)
    
    if len(lmList) >= 21:  # Ensure enough landmarks are detected
        # Thumb and index finger landmarks
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Draw landmarks and lines
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        # Calculate distance and adjust volume
        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [50, 300], [minvol, maxvol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        
        volume.SetMasterVolumeLevel(vol, None)
        
        # Change color if the distance is below threshold
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
    
    # Display volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Volume: {int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    
    # Display FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    cv2.imshow("Img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()