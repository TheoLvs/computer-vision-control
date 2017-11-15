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


from vision_control.utils import *



#=============================================================================================================================
# HELPER FUNCTIONS
#=============================================================================================================================



def predict(model,img,resize = None,crop = True,flatten = True):

    image = img.copy()

    if crop:
        image = crop_face(image)

    if resize:
        image = Image.fromarray(image).resize(resize)
        image = np.array(image)

    if flatten:
        image = image.reshape(np.prod(image.shape))

    image = np.divide(image,255)
    image = np.expand_dims(image,axis = 0)

    return model.predict(image)
