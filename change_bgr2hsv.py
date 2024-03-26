# coding: utf-8
import cv2
import numpy as np

# insert the BGR color
color = np.uint8([[[161,166,160]]])
hsv_color = cv2.cvtColor(color,cv2.COLOR_RGB2HSV)
print hsv_color
