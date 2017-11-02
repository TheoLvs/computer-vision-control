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
# CAMERA IMAGE
#=============================================================================================================================



class CameraImage(object):
    def __init__(self,image = None,file_path = None,capture = False,tag = None,check = True):

        if capture:
            image = self.capture()

        if file_path is not None:
            image = Image.open(file_path)


        if check:
            try:
                image.load()
                self.ok = True
            except IOError:
                self.ok = False


        if check == False or self.ok:
            self.file_path = file_path
            self.img = image
            self.array = self.to_array()
            self.original_array = np.copy(self.array)
            self.tag = tag
        
    def to_array(self):
        return np.array(self.img)
    
    def set_array(self,array):
        self.array = array
        self.img = Image.fromarray(array)
        
    def _repr_png_(self):
        return self.img._repr_png_()


    def preprocess(self,canny_intensity = 30):

        # Basic preprocessing
        img = to_black_and_white(self.original_array)
        # img = gaussian_smooth(img)
        img = detect_edges(img,canny_intensity,canny_intensity*2)
        img = gaussian_smooth(img)

        # Resizing
        img = np.array(Image.fromarray(img).resize((320,240)))

        # Setting the array
        self.set_array(img)



    def capture(self):
        cap = cv2.VideoCapture(0)
        _,img = cap.read()
        return Image.fromarray(img)


    def predict(self,model):
        self.preprocess()
        img = self.array
        x = img.reshape(1,img.shape[0]*img.shape[1])
        x = np.divide(x,255)
        prediction = model.predict(x)[0][0]
        return prediction





#=============================================================================================================================
# CAMERA IMAGES
#=============================================================================================================================


class CameraImages(object):
    def __init__(self,file_paths = None,camera_images = None):

        if file_paths is not None:
            self.images = [CameraImage(file_path = file_path) for file_path in file_paths]
        else:
            self.images = camera_images


    def __getitem__(self,key):
        return self.images[key]


    def preprocess(self):
        for image in tqdm(self.images,desc = "Preprocessing images"):
            image.preprocess()


    def build_X(self,flatten = True):
        
        for i,image in enumerate(self.images):
            img = image.array
            if flatten: 
                img = img.reshape(-1,img.shape[0]*img.shape[1])
            else:
                img = np.expand_dims(img,axis = 0)
                img = img.reshape((*img.shape,1))

            if i == 0:
                X = img
            else:
                X = np.vstack([X,img])

        return X



    def build_y(self):
        y = np.array([image.tag for image in self.images]).reshape(-1,1)
        return y

