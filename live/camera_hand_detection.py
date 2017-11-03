import numpy as np
import cv2
from PIL import Image

import sys
sys.path.append("../")

from vision_control.camera import *
from keras.models import model_from_json


print(">> Loading DL model")
model = model_from_json(open("C:/git/computer-vision-control/models/model_mlp.json","r").read())
model.load_weights("C:/git/computer-vision-control/models/model_mlp.h5")
print("ok")


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Getting image
    img = Image.fromarray(frame)

    # Image ontology
    image = CameraImage(image = img)

    # Prediction
    proba = float(image.predict(model))

    # Hand detection
    if proba > 0.5:
        print("HAND !")

    # Preprocessing
    edges = to_black_and_white(frame)
    edges = gaussian_smooth(edges)
    edges = detect_edges(edges,30,60)
    edges = gaussian_smooth(edges)

    # Put the probability in the frame
    cv2.putText(frame,"Hand probability : {:3g}".format(proba), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('edges',edges)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()