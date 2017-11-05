import numpy as np
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)
background = True

if background:
    fgbg = cv2.createBackgroundSubtractorMOG2()

i = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if background:
        frame = fgbg.apply(frame)

    Image.fromarray(frame).save("image_capture_{}.png".format(i))

    i += 1

    # Display the resulting frame
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()