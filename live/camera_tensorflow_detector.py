import numpy as np
import cv2
from ekimetrics.computer_vision.preprocessing import *
import tensorflow as tf


# https://github.com/datitran/object_detector_app/blob/master/object_detection_multithreading.py

print(">> LOADING TENSORFLOW MODEL")
from ekimetrics.computer_vision import object_detection as od
model_path = "C:/git/models/object_detection/coco/frozen_inference_graph.pb"
categories_path = "C:/git/models/object_detection/coco/categories.json"
detection_graph = od.load_frozen_tf_model(model_path)
categories = od.load_categories_dictionary(categories_path)
print("OK")

cap = cv2.VideoCapture(0)

while(True):

    with detection_graph.as_default():

        # CREATE THE TENSORFLOW SESSION
        with tf.Session(graph=detection_graph) as sess:

            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img = Image.fromarray(frame)
            img_predict = od.predict_live(sess,detection_graph,categories,img)

            # edges = to_black_and_white(frame)
            # edges = detect_edges(edges,50,100)

            # Display the resulting frame
            # cv2.imshow('frame',frame)
            # cv2.imshow('edges',edges)
            cv2.imshow('prediction',np.array(img_predict))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()