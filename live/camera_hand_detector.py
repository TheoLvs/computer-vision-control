#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
FACE DETECTOR LIVE TESTS
Started on the 09/11/2017
------------------------------------------------------------------------
"""




import numpy as np
import cv2
import datetime
import time
import sys

sys.path.append("..")

from vision_control import utils
from vision_control import face_id



#=============================================================================================================================
# MODELS
#=============================================================================================================================


# Load Cascade Open CV Classifiers
hand_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade1/aGest.xml')
hand_cascade2 = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade1/closed_frontal_palm.xml')
hand_cascade3 = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade1/palm.xml')
hand_cascade4 = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade1/fist.xml')
hand_cascade4 = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade2/Hand.Cascade.1.xml')


#=============================================================================================================================
# MAIN LOOP
#=============================================================================================================================


cap = cv2.VideoCapture(0)
smiling = False

while(True):

    # Capture frame-by-frame
    ret, img = cap.read()

    # Detect the hand
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade4.detectMultiScale(gray, 1.3, 5)

    # Iterate on the different faces detected
    for (x,y,w,h) in hands:


        # Select the region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(img,header, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))




    # # Attempt to detect the hands
    # hands = hand_cascade.detectMultiScale(gray)
    # for (x,y,w,h) in hands:
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(128,255,0),2)


    # Display the resulting frame
    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()