import cvzone
from test2 import handDetector

import cv2

cap = cv2.VideoCapture(0)

detector = handDetector()

while True:
    success , img = cap.read()
    colo = detector.findHands(img)
    
    # lmLIst = detector.findPosition(colo)
    lmLIst,bbox = detector.findPosition(img)
    
    cv2.imshow("image",img)
    cv2.waitKey(1)