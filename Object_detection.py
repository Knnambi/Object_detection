import numpy as np
import cv2
from imageai.Detection import ObjectDetection
import os
from PIL import ImageGrab

#fourcc = cv2.VideoWriter_fourcc('X','V','I','D') #you can use other codecs as well.
#vid = cv2.VideoWriter('record.avi', fourcc, 8, (500,490))
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
import math
time.sleep(5)
import os
while(True):
    img1 = ImageGrab.grab(bbox=(0,200,800,400)) #x, y, w, h
    img1.save("img1.jpg", "JPEG")
    detections = detector.detectObjectsFromImage(input_image='img1.jpg', output_image_path='combo.jpg', minimum_percentage_probability=30)
    img = cv2.imread('combo.jpg')
    cv2.imshow("frame", img)
    key = cv2.waitKey(1)
    if key == 27:
        break    

#vid.release()
cv2.destroyAllWindows()