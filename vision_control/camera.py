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
    def __init__(self,image = None,capture = False):

        if capture:
            image = self.capture()

        self.img = image
        self.array = self.to_array()
        self.original_array = np.copy(self.array)
        
    def to_array(self):
        return np.array(self.img)
    
    def set_array(self,array):
        self.array = array
        self.img = Image.fromarray(array)
        
    def _repr_png_(self):
        return self.img._repr_png_()


    def preprocess(self,canny_intensity = 50):

        # Basic preprocessing
        img = to_black_and_white(self.original_array)
        img = detect_edges(img,canny_intensity,canny_intensity*2)

        self.set_array(img)


    def capture(self):
        pass
        

