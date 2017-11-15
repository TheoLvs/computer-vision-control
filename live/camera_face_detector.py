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
face_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_smile.xml')
hand_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade1/aGest.xml')


# Load Face ID detection
model = utils.reload_keras_model("C:/git/computer-vision-control/models/face_fc_weights.h5","C:/git/computer-vision-control/models/face_fc_weights.json")


def format_prediction(prediction,classes_dict = None,force_argmax = None):
    argmax = np.argmax(np.squeeze(prediction))
    proba = np.squeeze(prediction)[argmax]*100
    if classes_dict is not None and (force_argmax is not None and force_argmax == argmax):
        class_ = classes_dict[argmax]
        return argmax,"{} : {:.1f}%".format(class_,proba)
    else:
        return argmax,"{:.1f}%".format(proba)


classes_dict = {0: 'dantec', 1: 'gus', 2: 'momo', 3: 'santi', 4: 'theo', 5: 'thib'}

#=============================================================================================================================
# MAIN LOOP
#=============================================================================================================================


cap = cv2.VideoCapture(0)
smiling = False

while(True):

    # Capture frame-by-frame
    ret, img = cap.read()

    # Detect the faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate on the different faces detected
    for (x,y,w,h) in faces:

        # Draw the boundaries of the faces

        # Select the region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # In the face ROI (Region Of Interest), find the eyes and draw them
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        # In the same ROI, find a smile
        smiles = smile_cascade.detectMultiScale(roi_gray,scaleFactor = 1.7,minNeighbors = 22)
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

        # Compute the time spent smiling
        if len(smiles) > 0:
            if not smiling:
                s = time.time()
                smiling = True
        else:
            if smiling:
                e = time.time()
                print("You smiled for {}s".format(e-s))
                smiling = False



        # Face ID
        prediction = face_id.predict(model,roi_color,resize = (100,100),flatten = True,crop = False)
        argmax,header = format_prediction(prediction,classes_dict,force_argmax = 4)

        if argmax == 4: # Class representing me
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,header, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,header, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))




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