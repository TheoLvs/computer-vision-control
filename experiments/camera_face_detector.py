import numpy as np
import cv2
import datetime
import time


face_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_smile.xml')


cap = cv2.VideoCapture(0)
smiling = False

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        smiles = smile_cascade.detectMultiScale(roi_gray,scaleFactor = 1.7,minNeighbors = 22)
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)


        if len(smiles) > 0:
            if not smiling:
                s = time.time()
                smiling = True
        else:
            if smiling:
                e = time.time()
                print("You smiled for {}s".format(e-s))
                smiling = False




    # Display the resulting frame
    cv2.imshow('frame',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()