#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
CAMERA
Started on the 01/11/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm




#=============================================================================================================================
# HELPER FUNCTIONS
#=============================================================================================================================


def to_black_and_white(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def detect_edges(img,threshold1 = 200,threshold2 = 300):
    return cv2.Canny(img, threshold1 = threshold1, threshold2=threshold2)


def gaussian_smooth(img):
    return cv2.GaussianBlur(img,(5,5),0)




#=============================================================================================================================
# HELPER FUNCTIONS
#=============================================================================================================================


def reload_keras_model(h5_path,json_path):
    from keras.models import model_from_json
    model = model_from_json(open(json_path,"r").read())
    model.load_weights(h5_path)
    return model


def save_keras_model(model,h5_path,json_path):
    model.save(h5_path)
    with open(json_path, "w") as json_file:
        json_file.write(model.to_json())





#=============================================================================================================================
# FACE DETECTION
#=============================================================================================================================



def detect_face(img,cascade_classifier = None):
    if cascade_classifier is None:
        cascade_classifier = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_frontalface_default.xml')

    faces = cascade_classifier.detectMultiScale(img, 1.3, 5)

    return faces




def crop_face(img,cascade_classifier = None):
    
    faces = detect_face(img,cascade_classifier = cascade_classifier)

    if len(faces) > 1:
        print("Warning : multiple faces")

    x,y,w,h = faces[0]

    return img[y:y+h, x:x+w]




def draw_face_contours(img,cascade_classifier = None,header = None,color = None):
    faces = detect_face(img,cascade_classifier = cascade_classifier)

    color = map_colors(color)

    if len(faces) > 1:
        print("Warning : multiple faces")

    new_img = img.copy()


    for (x,y,w,h) in faces:
        cv2.rectangle(new_img,(x,y),(x+w,y+h),color,2)

        if header is not None:
            cv2.putText(new_img,header, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color)


    return new_img



def map_colors(color = None):
    if color is None or color == "red":
        color = (255,0,0)
    elif color == "green":
        color = (0,255,0)
    elif color == "blue":
        color = (0,0,255)
    else:
        pass

    return color




