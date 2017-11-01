import numpy as np
import cv2


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)


    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

