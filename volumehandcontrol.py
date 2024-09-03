import cv2
import time
import math
import numpy as np
import handtrackmodule as htm
import subprocess

wCam,hCam=640,600

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
ptime=0
detector=htm.handDetector(detectionCon=0.7)
minVol=0
maxVol=100
volBar=400
volPer=0
while True:
    success,img=cap.read()
    if not success:
        print("Failed to capture image from camera.")
        break
    img=detector.findHands(img)
    lmList,bbox=detector.findPosition(img,draw=False)
    num_landmarks=len(lmList)
    # cv2.putText(img, f'Landmarks Detected: {num_landmarks}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    if len(lmList)>=21:
        try:
            #Extract landmarks for thumb and index finger
            x1,y1=lmList[4][1],lmList[4][2]  #Thumb tip
            x2,y2=lmList[8][1],lmList[8][2]  #Index finger tip
            cx,cy=(x1+x2)//2,(y1+y2)//2
            length=math.hypot(x2-x1,y2-y1)
            vol=np.interp(length,[50,200],[minVol,maxVol])
            volBar=np.interp(length,[50,200],[400,150])
            volPer=np.interp(length,[50,200],[0,100])  
            #Set system volume using pactl (Linux specific command)
            subprocess.run(['pactl','set-sink-volume','@DEFAULT_SINK@',f'{int(vol)}%'])
            print(f'Volume: {int(volPer)}%')  
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)  #Ensure correct radius
            cv2.circle(img,(x2,y2),15,(255,0,255),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)  #Create a line between the points
            cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
            cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
            cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),-1)  #start position, end position,color,thickness
            cv2.putText(img,f'Volume: {int(volPer)}%',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            ctime=time.time()
            fps=1/(ctime-ptime)
            ptime=ctime
            cv2.putText(img,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        except IndexError as e:
            print(f"Index error while accessing lmList elements: {e}")
    else:
        volBar=400  #Default bar position when no hand detected
        volPer=0  #Default percentage when no hand detected
    cv2.imshow("Img",img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
