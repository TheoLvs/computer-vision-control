import numpy as np
import cv2
from ekimetrics.computer_vision.preprocessing import *


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    edges = to_black_and_white(frame)
    edges = detect_edges(edges,50,100)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('edges',edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()