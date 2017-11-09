import numpy as np
import cv2
from PIL import Image
import time
from sklearn.cluster import MeanShift

background_reduction = False
ms = MeanShift(bandwidth=20)

fgbg = cv2.createBackgroundSubtractorMOG2()

def preprocess_and_show(img,hvs = (107,175,70),background_reduction = background_reduction):

    if background_reduction:
        x = fgbg.apply(img)
        x_original = img.copy()
        x_preprocessed = x.copy()

    else:
        x = img[:,:,[2,1,0]]
        x_original = x[:,:,[2,1,0]].copy()

        # To HSV colormap
        x = cv2.cvtColor(x,cv2.COLOR_BGR2HSV)

        # Filter out some color ranges
        x = cv2.inRange(x,(107, 70,0),(255,255,255))

        x_preprocessed = None



    # Smooth the image with a median blur
    x = cv2.medianBlur(x,5)

    # Dilate to fill the holes
    element_size = 5
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * element_size + 1, 2 * element_size + 1), (element_size, element_size))
    x = cv2.dilate(x,element)

    # Find the contours
    x, contours, hierarchy = cv2.findContours(x,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        max_contour = np.argmax([y.shape[0] for y in contours])

        # Draw contours
        cv2.drawContours(x_original,contours,max_contour,(0,255,0),2)

        # Hull convex detection
        hull = [cv2.convexHull(contours[max_contour])]
        cv2.drawContours(x_original,hull,0,(0,0,255),2)

    return Image.fromarray(x_original),x_preprocessed,hull[0]


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    # Preprocessing
    img,x_preprocessed,hull = preprocess_and_show(frame)
    img = np.array(img)

    # Find number of fingers
    n_angles = len(np.bincount(ms.fit_predict(hull.reshape(-1,2))))
    print(n_angles)

    # Display the resulting frame
    cv2.imshow('frame',img)

    if background_reduction:
        cv2.imshow('frame2',x_preprocessed)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()